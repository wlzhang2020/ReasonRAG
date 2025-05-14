import argparse
import json
import time
from multiprocessing import Process
import copy
import os
from tqdm import tqdm
import yaml
from flashrag.config import Config
from flashrag.utils import get_dataset
from pipeline.reasonrag_pipeline import ReasonRAGPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--model", type=str)
args = parser.parse_args()


root_dir = 'output'

def load_config_from_yaml(yaml_file):
    try:
        with open(yaml_file, "r") as file:
            return yaml.safe_load(file)
    except Exception as e:
        print(f"Error loading YAML file: {e}")
        return {}

default_config = load_config_from_yaml("my_config.yaml")

config_dict = {
    "data_dir": "dataset/",
    "dataset_name": args.dataset_name,
    "index_path": "indexes/bge_Flat_wiki_extend.index",
    "corpus_path": "indexes/wiki18_100w_extend.jsonl",
    "model2path": {
        "bge": "BAAI/bge-base-en-v1.5",
        "qwen2.5": "Qwen/Qwen2.5-7B-Instruct",
    },
    "generator_model": "gpt-4o-2024-05-13",
    "retrieval_method": "bge",
    "framework": "openai",
    "gpu_id": "1",
    "faiss_gpu": True,
    "metrics": ["em", "f1", "acc", "recall", "precision"],
    "retrieval_topk": 3,
    "save_intermediate_data": True,
    "save_note": args.model + "_MCTS",
}

answer_format = "answer"
max_iter = 8

config_dict = {**default_config, **config_dict}
config = Config(config_dict=config_dict)

dataset_path = config["dataset_path"]
split_path = os.path.join(dataset_path, "train.jsonl")
data_split = "train"
all_split = get_dataset(config)
test_data = all_split[data_split]

def save_data(save_dir, data, file_name="intermediate_data.json"):
    data = [item.to_dict() for item in data]
    save_path = os.path.join(save_dir, file_name)
    with open(save_path, "w", encoding="utf-8") as f:
        json.dump(data, f, indent=4)


def parallel_process_dataset(config, test_data, num_processes=4):
    total_items = len(test_data)
    if total_items == 0:
        print("No data to process.")
        return

    chunk_size = total_items // num_processes
    remainder = total_items % num_processes

    chunks = []
    start = 0
    for i in range(num_processes):
        end = start + chunk_size
        if i < remainder:
            end += 1
        if start >= total_items:
            break
        chunks.append(test_data[start:end])
        start = end

    processes = []
    for chunk_idx, chunk in enumerate(chunks):
        print(f"Chunk {chunk_idx} ready.")
        p = Process(target=process_chunk, args=(copy.copy(config), chunk, chunk_idx))
        processes.append(p)
        p.start()

    for p in processes:
        p.join()


def process_chunk(config, chunk, chunk_idx):
    save_dir = os.path.join(config["save_dir"], f"chunk_{chunk_idx}")
    os.makedirs(save_dir, exist_ok=True)
    config["save_dir"] = save_dir

    pipeline = ReasonRAGPipeline(config, prompt_template=None, max_iter=7, max_children=2, max_rollouts=64)

    i = 0
    start_time = time.time()

    for item in tqdm(chunk, desc=f"Chunk {chunk_idx}"):
        try:
            pipeline.search(item)
        except Exception as e:
            print(f"Chunk {chunk_idx} Error at item {i}: {e}")
            continue
        finally:
            i += 1

    save_data(save_dir, chunk, file_name=f"final_{chunk_idx}.json")
    print(f"Chunk {chunk_idx} processed {len(chunk)} items in {time.time() - start_time:.2f}s")


parallel_process_dataset(config, test_data, num_processes=10)
