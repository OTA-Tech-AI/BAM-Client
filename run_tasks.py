import argparse
import asyncio
import json
import logging
import os
import random
import shutil
import string
from asyncio import Semaphore
from dataclasses import dataclass, field
from pathlib import Path
from typing import Generator, List, Literal, Set, TypedDict

from browser_use import Agent, Browser, BrowserConfig
from browser_use.browser.context import BrowserContextConfig
from dotenv import load_dotenv
from langchain_anthropic import ChatAnthropic
from langchain_openai import AzureChatOpenAI, ChatOpenAI
from langchain_ollama import ChatOllama
from pydantic import SecretStr
from browser_use.agent.next_goal_saver import init_global_next_goal_saver_list

load_dotenv()

GLOBAL_CONCURRENT_AGENT_ID = []
global_id_lock = asyncio.Lock()

class TaskData(TypedDict):
    id: str
    web: str
    ques: str

EvalResult = Literal["success", "failed", "unknown"]

@dataclass
class RunStats:
    total_tasks: int
    current_task: int = 0
    successful_tasks: Set[str] = field(default_factory=set)
    failed_tasks: Set[str] = field(default_factory=set)
    unknown_tasks: Set[str] = field(default_factory=set)

    def update(self, task_id: str, success: "EvalResult") -> None:
        if success == "success":
            self.successful_tasks.add(task_id)
        elif success == "failed":
            self.failed_tasks.add(task_id)
        else:
            self.unknown_tasks.add(task_id)


def cleanup_webdriver_cache() -> None:
    """Clean up webdriver cache directories."""
    cache_paths = [
        Path.home() / ".wdm",
        Path.home() / ".cache" / "selenium",
        Path.home() / "Library" / "Caches" / "selenium",
    ]
    for path in cache_paths:
        if path.exists():
            print(f"Removing cache directory: {path}")
            shutil.rmtree(path, ignore_errors=True)


def generate_random_agent_ids(max_concurrent_num: int) -> None:
    for _ in range(max_concurrent_num):
        random_id = ''.join(random.choices(string.ascii_letters + string.digits, k=8))
        GLOBAL_CONCURRENT_AGENT_ID.append(random_id)

async def pop_concurrent_agent_id() -> str:
    async with global_id_lock:
        if GLOBAL_CONCURRENT_AGENT_ID:
            return GLOBAL_CONCURRENT_AGENT_ID.pop()
        else:
            raise ValueError("No more agent IDs available.")

async def append_agent_id(agent_id: str) -> None:
    async with global_id_lock:
        GLOBAL_CONCURRENT_AGENT_ID.append(agent_id)

@dataclass
class LLMModel:
    model: AzureChatOpenAI
    token_limit: int

def get_llm_model_generator(
    model_provider: str,
) -> Generator[AzureChatOpenAI | ChatAnthropic | ChatOpenAI, None, None]:
    """Generator that creates fresh model instances each time"""
    while True:
        # Force reload environment variables
        load_dotenv(override=True)

        if model_provider == "anthropic":
            # Create fresh instances each time, reading current env vars
            yield ChatAnthropic(
                model_name="claude-3-7-sonnet-20250219",
                timeout=25,
                stop=None,
                temperature=0.0,
            )

        elif model_provider == "azure":
            # Create fresh instances each time, reading current env vars
            west_eu = LLMModel(
                model=AzureChatOpenAI(
                    model="gpt-4o",
                    api_version="2024-10-21",
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_WEST_EU", ""),
                    api_key=SecretStr(os.getenv("AZURE_OPENAI_API_KEY_WEST_EU", "")),
                ),
                token_limit=900,
            )
            east_us = LLMModel(
                model=AzureChatOpenAI(
                    model="gpt-4o",
                    api_version="2024-10-21",
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_EAST_US", ""),
                    api_key=SecretStr(os.getenv("AZURE_OPENAI_API_KEY_EAST_US", "")),
                ),
                token_limit=450,
            )
            east_us_2 = LLMModel(
                model=AzureChatOpenAI(
                    model="gpt-4o",
                    api_version="2024-10-21",
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_EAST_US_2", ""),
                    api_key=SecretStr(os.getenv("AZURE_OPENAI_API_KEY_EAST_US_2", "")),
                ),
                token_limit=450,
            )
            west_us = LLMModel(
                model=AzureChatOpenAI(
                    model="gpt-4o",
                    api_version="2024-10-21",
                    azure_endpoint=os.getenv("AZURE_OPENAI_ENDPOINT_WEST_US", ""),
                    api_key=SecretStr(os.getenv("AZURE_OPENAI_API_KEY_WEST_US", "")),
                ),
                token_limit=450,
            )

            # Yield fresh instances in the same pattern
            yield west_eu.model  # First 900
            yield west_eu.model  # Second 900
            yield east_us.model  # 450
            yield east_us_2.model  # 450
            yield west_us.model  # 450
        elif model_provider == "openai":
            llm = ChatOpenAI(model="gpt-4o", temperature=0)
            yield llm
        elif model_provider == "ollama":
            llm = ChatOllama(model="OTA-v1", num_ctx=20000,temperature=0)
            yield llm
        else:
            raise ValueError(f"Invalid model provider: {model_provider}")


async def process_single_task(
    task: TaskData,
    client: AzureChatOpenAI | ChatAnthropic | ChatOpenAI,
    stats: RunStats,
    results_dir: Path,
    browser: Browser,
) -> None:
    """Process a single task asynchronously."""
    task_str = f"{task['ques']} on {task['web']}"
    task_dir = results_dir / f"{task['id']}"
    task_dir.mkdir(exist_ok=True)
    
    task_concurrent_agent_id = await pop_concurrent_agent_id()

    try:
        if not (task_dir / "task_result.json").exists():
            logging.getLogger("browser_use").setLevel(logging.INFO)

            agent = Agent(
                task=task_str,
                llm=client,
                browser=browser,
                validate_output=True,
                generate_gif=False,
                use_vision=False,
                concurrent_agent_id=task_concurrent_agent_id,
            )
            history = await agent.run(max_steps=20)
            history.save_to_file(task_dir / "history.json")

    except Exception as e:
        logging.error(f"Error processing task {task['id']}: {str(e)}")
        stats.update(task["id"], "failed")  # Mark as failed instead of crashing
        return

    finally:
        await append_agent_id(task_concurrent_agent_id)
        await browser.close()


async def main(max_concurrent_tasks: int, model_provider: str, task_jsonl_path: str) -> None:
    try:
        # Setup
        cleanup_webdriver_cache()
        semaphore = Semaphore(max_concurrent_tasks)

        # Load tasks
        tasks: List[TaskData] = []
        with open(task_jsonl_path, "r") as f:
            for line in f:
                tasks.append(json.loads(line))

        # remove impossible tasks
        with open("testcases/WebVoyagerImpossibleTasks.json", "r") as f:
            impossible_tasks = set(json.load(f))
        tasks = [task for task in tasks if task["id"] not in impossible_tasks]

        # Initialize
        stats = RunStats(total_tasks=len(tasks))
        results_dir = Path("results")
        results_dir.mkdir(parents=True, exist_ok=True)

        # Process tasks concurrently with semaphore
        async def process_with_semaphore(
            task: TaskData, client: AzureChatOpenAI | ChatAnthropic | ChatOpenAI,
        ) -> None:
            async with semaphore:
                print(f"\n=== Now at task {task['id']} ===")

                # Create browser instance inside the semaphore block
                browser = Browser(
                    config=BrowserConfig(
                        # headless=True,
                        headless=False,
                        disable_security=True,
                        new_context_config=BrowserContextConfig(
                            disable_security=True,
                            wait_for_network_idle_page_load_time=5,
                            maximum_wait_page_load_time=20,
                            # no_viewport=True,
                            browser_window_size={
                                "width": 1280,
                                "height": 1100,
                            },
                        ),
                    )
                )
                await process_single_task(
                    task,
                    client,
                    stats,
                    results_dir,
                    browser,  # Pass browser instance
                )
                stats.current_task += 1

                # Add this to ensure browser is always closed
                try:
                    await browser.close()
                except Exception as e:
                    logging.error(f"Error closing browser: {e}")

                print(f"Current task: {stats.current_task}")
                print(f"Total tasks: {stats.total_tasks}")

        # Create and run all tasks
        all_tasks = []
        for i, task in enumerate(tasks):
            model = next(get_llm_model_generator(model_provider))
            all_tasks.append(process_with_semaphore(task, model))

        # Add timeout and better error handling
        await asyncio.gather(*all_tasks, return_exceptions=True)
    except Exception as e:
        logging.error(f"Main loop error: {e}")
    finally:
        # Cleanup code here
        logging.info("Shutting down...")


if __name__ == "__main__":
    if os.path.exists("results"):
        shutil.rmtree("results")
    try:
        parser = argparse.ArgumentParser(
            description="Run browser tasks with concurrent execution"
        )
        parser.add_argument(
            "--max-concurrent",
            type=int,
            default=1,
            help="Maximum number of concurrent tasks (default: 1)",
        )
        parser.add_argument(
            "--model-provider",
            type=str,
            default="azure",
            help="Model provider (default: azure)",
            choices=[
                "azure",
                "anthropic",
                "openai",
                "ollama",
            ],
        )
        parser.add_argument(
            "--task_jsonl_path",
            type=str,
            default="testcases/OTA_testdataset_mini.jsonl",
            help="the jsonl file for reading tasks",
        )

        args = parser.parse_args()
        logging.info(f"Running with {args.max_concurrent} concurrent tasks")
        generate_random_agent_ids(args.max_concurrent)
        init_global_next_goal_saver_list(GLOBAL_CONCURRENT_AGENT_ID)

        asyncio.run(main(args.max_concurrent, args.model_provider, args.task_jsonl_path))
    except KeyboardInterrupt:
        print("\nReceived keyboard interrupt, shutting down...")
    except Exception as e:
        print(f"Fatal error: {e}")
        logging.exception("Fatal error occurred")
