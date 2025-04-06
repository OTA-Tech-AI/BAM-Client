from typing import List

GLOBAL_NEXT_GOAL_SAVER_LIST = {}
class browserUseNextGoalSaver:

    def __init__(self, concurrent_agent_id):
        self.concurrent_agent_id = concurrent_agent_id
        self.next_goal = ""
    
    def set_next_goal(self, next_goal: str) -> None:
        self.next_goal = next_goal


def init_global_next_goal_saver_list(agent_ids: List[str]):
    global GLOBAL_NEXT_GOAL_SAVER_LIST
    for agent_id in agent_ids:
        # Optionally, you can pass the agent_id into the constructor if required.
        GLOBAL_NEXT_GOAL_SAVER_LIST[agent_id] = browserUseNextGoalSaver(agent_id)


def reset_global_next_goal_saver(agent_id: str):
    global GLOBAL_NEXT_GOAL_SAVER_LIST

    # Remove the existing goal saver if it exists.
    if agent_id in GLOBAL_NEXT_GOAL_SAVER_LIST:
        del GLOBAL_NEXT_GOAL_SAVER_LIST[agent_id]

    # Create a new instance of browserUseNextGoalSaver for the agent.
    GLOBAL_NEXT_GOAL_SAVER_LIST[agent_id] = browserUseNextGoalSaver(agent_id)
