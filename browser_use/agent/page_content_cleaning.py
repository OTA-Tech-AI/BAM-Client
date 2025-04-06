import re
from sentence_transformers import SentenceTransformer
from typing import List, Tuple
from langchain_core.messages import (AIMessage, BaseMessage)
from browser_use.agent.next_goal_saver import GLOBAL_NEXT_GOAL_SAVER_LIST
import torch

REMOVE_NEWLINES = False
LINE_CHAR_LIMIT = 200
TOP_K_ELEMENTS = 200
LAST_N_ACTION_HISTORY = 2

# Set device to GPU if available, else CPU
device = torch.device("cuda" if torch.cuda.is_available() else "cpu")
# Initialize the model and move to appropriate device
embedding_model = SentenceTransformer('intfloat/multilingual-e5-large-instruct').to(device)


def similarity_search(task: str, query: str, documents: List[str], top_k: int = 3) -> List[Tuple[str, float]]:
    # Format the query with task instruction
    def get_detailed_instruct(task_description: str, query: str) -> str:
        return f'Instruct: {task_description}\nQuery: {query}'
    
    formatted_query = get_detailed_instruct(task, query)
    
    # Prepare input texts (query + documents)
    input_texts = [formatted_query] + documents
    
    # Generate embeddings on the selected device
    embeddings = embedding_model.encode(input_texts, 
                            convert_to_tensor=True, 
                            normalize_embeddings=True,
                            device=device)
    
    # Calculate similarity scores
    scores = (embeddings[0] @ embeddings[1:].T) * 100
    
    # Convert to list and pair with documents
    scores_list = scores.tolist()
    document_score_pairs = list(zip(documents, scores_list))
    
    # Sort by score in descending order and get top_k
    sorted_pairs = sorted(document_score_pairs, key=lambda x: x[1], reverse=True)
    top_results = sorted_pairs[:min(top_k, len(documents))]
    
    return top_results


def sort_similarity_results_by_index(results: List[Tuple[str, float]]) -> List[Tuple[str, float]]:
    def get_index(item: Tuple[str, float]) -> int:
        doc, score = item
        match = re.match(r"\[(\d+)\]", doc)
        return int(match.group(1)) if match else 0
    return sorted(results, key=get_index)


def join_index_blocks(text: str) -> str:
    # This regex finds a group consisting of an index marker and the subsequent content,
    # until the next index marker or end-of-text.
    pattern = re.compile(r"(\[\d+\])(.*?)(?=\n\[\d+\]|$)", re.DOTALL)
    
    blocks = pattern.findall(text)
    joined_blocks = []
    for idx, content in blocks:
        # Split the content into lines, remove extra whitespace, and join with a space.
        joined_content = " ".join(content.splitlines()).strip()
        joined_blocks.append(f"{idx}{joined_content}")
    
    # Reassemble all blocks with a newline separator.
    return "\n".join(joined_blocks)


def cleanup_page_symbol_content(raw_text, concurrent_agent_id):
    preserve_count = 10  # Adjust this value as needed
    print(f"[OTA]: start to cleanup page content, raw text length: {len(raw_text)}")
    # Remove all <img /> tags
    cleaned_text = re.sub(r"<img\s*/>", "", raw_text)

    # Replace literal "\n" with a space
    cleaned_text = cleaned_text.replace("\\n", " ")

    # Add newlines before each index like [0], [1], etc.
    cleaned_text = re.sub(r"(?<!\n)(\[\d+\])", r"\n\1", cleaned_text)

    # Remove index lines like: [123] (with nothing after it)
    cleaned_text = re.sub(r"\n\[\d+\]\s*(?=\n|\Z)", "", cleaned_text)

    # Remove attributes from <a> and <iframe> if there's content
    def clean_tag(match):
        tag, content = match.group(1), match.group(2)
        return f"<{tag}>{content}/>" if content.strip() else match.group(0)
    cleaned_text = re.sub(r"<(a|iframe)[^>]*?>([^<>]+?)/>", clean_tag, cleaned_text)

    # Remove empty tags like <a />, <div />, etc. that carry no content
    cleaned_text = re.sub(r"\[\d+\]<\w+\s*/>", "", cleaned_text)

    cleaned_lines = cleaned_text.splitlines()
    cleaned_lines = [line.strip() for line in cleaned_lines]
    cleaned_text = "\n".join(cleaned_lines)

    # Remove input tags like [15]<input checkbox/> or <input radio/>
    cleaned_text = re.sub(r"\[\d+\]<input\s+(checkbox|radio|text|password|submit|reset)\s*/>", "", cleaned_text, flags=re.IGNORECASE)

    # Remove specific non-informative self-closing tags like <img presentation/>, <meta analytics/>, etc.
    non_informative_tags = ['img', 'meta']
    non_informative_types = ['presentation', 'preload', 'checkbox', 'radio', 'text']

    for tag in non_informative_tags:
        for tag_type in non_informative_types:
            pattern = rf"\[\d+\]<{tag}\s+{tag_type}\s*/>"
            cleaned_text = re.sub(pattern, "", cleaned_text, flags=re.IGNORECASE)

    cleaned_text = join_index_blocks(cleaned_text)

    # Hard cutoff each line to `line_char_limit` characters
    cleaned_lines = cleaned_text.splitlines()
    trimmed_lines = []
    for line in cleaned_lines:
        line = line.strip()
        if not line:
            continue
        match = re.match(r"^(\[\d+\])(.+)$", line)
        if match:
            prefix, content = match.group(1), match.group(2)
            if len(content) > LINE_CHAR_LIMIT:
                content = content[:LINE_CHAR_LIMIT] + "..."
            trimmed_lines.append(f"{prefix}{content}")
        else:
            trimmed_lines.append(line)

    if concurrent_agent_id != "" and GLOBAL_NEXT_GOAL_SAVER_LIST[concurrent_agent_id].next_goal != "":
        if len(trimmed_lines) > TOP_K_ELEMENTS:
            # Local variable to adjust the number of preserved indexed elements

            # Split into preserved top `preserve_count` and the rest
            indexed_lines = [(line, int(re.match(r"^\[(\d+)\]", line).group(1))) 
                           for line in trimmed_lines if re.match(r"^\[\d+\]", line)]
            non_indexed_lines = [line for line in trimmed_lines if not re.match(r"^\[\d+\]", line)]
            
            # Sort by index and get top `preserve_count`
            indexed_lines.sort(key=lambda x: x[1])  # Sort by index number
            top_preserved = [line for line, _ in indexed_lines[:preserve_count]]
            remaining_indexed = [line for line, _ in indexed_lines[preserve_count:]]

            # Apply similarity search only to remaining indexed lines
            if remaining_indexed:
                top_results = similarity_search(
                    task="Given a web action query, retrieve relevant elements that is related to the query",
                    query=GLOBAL_NEXT_GOAL_SAVER_LIST[concurrent_agent_id].next_goal,
                    documents=remaining_indexed,
                    top_k=TOP_K_ELEMENTS - len(top_preserved)  # Adjust top_k to account for preserved lines
                )
                sorted_results = sort_similarity_results_by_index(top_results)
                sorted_lines = [doc for doc, score in sorted_results]
            else:
                sorted_lines = []

            # Combine preserved top `preserve_count`, similarity results, and non-indexed lines
            trimmed_lines = top_preserved + sorted_lines + non_indexed_lines

    # Step 7: Join final output
    if REMOVE_NEWLINES:
        cleaned_text = " ".join(trimmed_lines)
    else:
        cleaned_text = "\n".join(trimmed_lines)

    print(f"[OTA]: cleanup page content finished, cleaned text length: {len(cleaned_text)}")
    return cleaned_text


def cleanup_task_history(messages: List[BaseMessage], concurrent_agent_id: str) -> List[BaseMessage]:

    start_marker = "[Your task history memory starts here]"
    end_marker = "[Task history memory ends]"
    
    start_idx = None
    end_idx = None

    for i, msg in enumerate(messages):
        if start_marker in msg.content:
            start_idx = i
        if end_marker in msg.content:
            end_idx = i
            # We assume the first occurrence of the end marker after the start marker is the correct one.
            if start_idx is not None:
                break

    if start_idx is None or end_idx is None or end_idx <= start_idx:
        return messages

    inner_messages = messages[start_idx + 1 : end_idx]
    ai_indices = [i for i, msg in enumerate(inner_messages) if isinstance(msg, AIMessage)]

    if not ai_indices:
        return messages
    elif len(ai_indices) < LAST_N_ACTION_HISTORY:
        last_group_start = 0
    else:
        last_group_start = ai_indices[-LAST_N_ACTION_HISTORY]

    kept_messages = inner_messages[last_group_start:]
    if last_group_start is None:
        return messages

    kept_messages = inner_messages[last_group_start:]
    new_messages = messages[:start_idx + 1] + kept_messages + messages[end_idx:]

    return new_messages