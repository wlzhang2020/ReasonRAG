import os
import json
from collections import defaultdict
from typing import Dict, List
import numpy as np
import re
import tiktoken
import string
from glob import glob

BETA = 0.9
ENCODER = tiktoken.get_encoding("cl100k_base")

def count_tokens(text: str) -> int:
    return len(ENCODER.encode(text))

def get_action_type(response: str) -> str:
    if "<query>" in response:
        return "Query Generation"
    elif "<evidence>" in response:
        return "Evidence Extraction"
    elif "<answer>" in response:
        return "Answer Generation"
    return "Other"

def get_content(response: str) -> str:
    for tag in ["answer", "query", "evidence"]:
        match = re.search(rf"<{tag}>(.*?)</{tag}>", response, re.DOTALL)
        if match:
            return match.group(1).strip()
    return ""

def get_action_type_and_content(response: str) -> tuple[str, str]:
    action_type = get_action_type(response)
    content = get_content(response)
    return action_type, content

def process_node(data: Dict, node_id: int, results: List, action_type_counts: Dict[str, int], layer: int = 0):
    current_node = data[f"intermediate_node_{node_id}"]
    children_ids = current_node.get("children_ids", [])
    if len(children_ids) < 2:
        return

    valid_children = []
    prompt = ""
    system_prompt = ""
    user_prompt = ""
    for child_id in children_ids:
        child_node = data.get(f"intermediate_node_{child_id}")
        if child_node and "reward" in child_node and child_node["reward"] is not None and "input_prompt" in child_node:
            valid_children.append(child_node)
            input_prompt = child_node["input_prompt"]
            system_prompt = input_prompt[0]["content"]
            user_prompt = input_prompt[1]["content"]
            prompt = f"System: {system_prompt}\nUser: {user_prompt}"

    for i in range(len(valid_children)):
        for j in range(i + 1, len(valid_children)):
            node_a, node_b = valid_children[i], valid_children[j]
            reward_a, reward_b = float(node_a["reward"]), float(node_b["reward"])
            if abs(reward_a - reward_b) < 0.01:
                continue

            chosen, rejected = (node_a, node_b) if reward_a > reward_b else (node_b, node_a)
            chosen_action_type, chosen_content = get_action_type_and_content(chosen["response"])
            rejected_action_type, rejected_content = get_action_type_and_content(rejected["response"])

            if chosen_content == rejected_content or chosen_action_type == "Other":
                continue

            action_type_counts[chosen_action_type] += 1
            results.append({
                "instruction": system_prompt,
                "input": user_prompt,
                "prompt": prompt,
                "chosen": chosen["response"],
                "rejected": rejected["response"],
                "chosen_action_type": chosen_action_type,
                "chosen_content": chosen_content,
                "rejected_action_type": rejected_action_type,
                "rejected_content": rejected_content,
                "layer": layer
            })

    for child in valid_children:
        process_node(data, child["id"], results, action_type_counts, layer + 1)

def process_json_files(folder_list: List[str], output_file: str = "dpo_data.json"):
    all_results = []
    action_type_counts = defaultdict(int)

    for folder in folder_list:
        pattern = os.path.join(folder, "chunk_*", "reward_*.json")
        for json_path in glob(pattern, recursive=True):
            try:
                with open(json_path, "r") as f:
                    items = json.load(f)
            except Exception:
                continue

            for item in items:
                if "output" not in item or "golden_answers" not in item or item["golden_answers"] is None:
                    continue

                output_data = item["output"]
                root_node = output_data.get("intermediate_node_0")
                if not root_node or root_node.get("reward") is None:
                    continue

                process_node(output_data, 0, all_results, action_type_counts)

    with open(output_file, "w") as f:
        json.dump(all_results, f, indent=2)

def normalize_answer(s: str) -> str:
    def white_space_fix(text):
        return " ".join(text.split())
    def remove_punc(text):
        exclude = set(string.punctuation)
        return "".join(ch for ch in text if ch not in exclude)
    def lower(text):
        return text.lower()
    return white_space_fix(remove_punc(lower(s))).strip()

def compute_f1_single(prediction: str, ground_truth: str) -> float:
    pred, gt = normalize_answer(prediction), normalize_answer(ground_truth)
    if pred in ["yes", "no", "noanswer"] and pred != gt:
        return 0.0
    if gt in ["yes", "no", "noanswer"] and pred != gt:
        return 0.0

    pred_tokens, gt_tokens = pred.split(), gt.split()
    common = sum((Counter(pred_tokens) & Counter(gt_tokens)).values())
    if common == 0:
        return 0.0
    precision = common / len(pred_tokens) if pred_tokens else 0.0
    recall = common / len(gt_tokens) if gt_tokens else 0.0
    return (2 * precision * recall) / (precision + recall) if (precision + recall) > 0 else 0.0

def compute_f1_multiple_gt(prediction: str, ground_truths: List[str]) -> float:
    return max(compute_f1_single(prediction, gt) for gt in ground_truths)

def calculate_mcts_reward_bottom_up(data: Dict, golden_answers: List[str]) -> Dict:
    unprocessed_children_count = {
        k: len(data[k].get("children_ids", [])) for k in data if "intermediate_node_" in k
    }
    queue = deque(k for k, count in unprocessed_children_count.items() if count == 0)
    rewards = {}
    updated_nodes = data.copy()

    while queue:
        node_key = queue.popleft()
        node = updated_nodes[node_key]

        if not node.get("children_ids"):
            reward = None
            if "answer" in node and node.get("response") and node["answer"] is not None:
                f1 = compute_f1_multiple_gt(node["answer"], golden_answers)
                step = node.get("step", 0)
                reward = f1 * (BETA ** step)
        else:
            children_keys = [f"intermediate_node_{cid}" for cid in node["children_ids"]]
            valid_children = [ck for ck in children_keys if ck in rewards and rewards[ck] is not None]
            reward = None
            if valid_children:
                total_n = sum(updated_nodes[ck].get("N", 0) for ck in valid_children)
                if total_n > 0:
                    reward = sum(rewards[ck] * updated_nodes[ck].get("N", 0) for ck in valid_children) / total_n

        rewards[node_key] = reward
        if reward is not None:
            updated_nodes[node_key]["reward"] = reward
        elif "reward" in updated_nodes[node_key]:
            del updated_nodes[node_key]["reward"]

        parent_id = node.get("parent_id")
        if parent_id is not None:
            parent_key = f"intermediate_node_{parent_id}"
            if parent_key in updated_nodes:
                unprocessed_children_count[parent_key] -= 1
                if unprocessed_children_count[parent_key] == 0:
                    queue.append(parent_key)

    return updated_nodes

def process_json_file_and_calculate_reward_save(json_path: str):
    try:
        with open(json_path, "r") as f:
            items = json.load(f)
            updated_items = []
            for item in items:
                if "output" not in item or item.get("golden_answers") is None:
                    updated_items.append(item)
                    continue

                mcts_data = item["output"]
                if "intermediate_node_0" not in mcts_data:
                    updated_items.append(item)
                    continue

                updated_nodes = calculate_mcts_reward_bottom_up(mcts_data, item["golden_answers"])
                modified_item = item.copy()
                modified_item["output"] = updated_nodes
                updated_items.append(modified_item)

        chunk_folder = os.path.dirname(json_path)
        base_name = os.path.splitext(os.path.basename(json_path))[0]
        sequence_number = base_name.split("_")[-1]
        reward_output_path = os.path.join(chunk_folder, f"reward_{sequence_number}.json")
        with open(reward_output_path, "w") as f:
            json.dump(updated_items, f, indent=2)

    except Exception:
        pass

def process_folder_and_calculate_rewards_save(folder_path: str):
    pattern = os.path.join(folder_path, "chunk_*", "intermediate_*.json")
    for json_path in glob(pattern, recursive=True):
        process_json_file_and_calculate_reward_save(json_path)

def main():
    folder_list = ["output/2wikimultihopqa_2025_02_04_01_49_experiment"]
    for folder in folder_list:
        process_folder_and_calculate_rewards_save(folder)
    process_json_files(folder_list, output_file="training_data/RAG_ProGuide.json")

if __name__ == "__main__":
    main()
