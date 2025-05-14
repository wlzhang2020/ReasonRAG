import argparse
import os
import yaml
from flashrag.config import Config
from flashrag.utils import get_dataset
from pipeline.reasonrag_pipeline import ReasonRAGPipeline

parser = argparse.ArgumentParser()
parser.add_argument("--dataset_name", type=str)
parser.add_argument("--model", type=str)
parser.add_argument("--max_iter", default=8, type=int)
parser.add_argument("--retrieval_top_k", default=3, type=int)
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
    "index_path": "indexes/bge_Flat.index",
    "retrieval_method": "bge",
    "corpus_path": "indexes/wiki18_100w.jsonl",
    "model2path": {
        "bge": "BAAI/bge-base-en-v1.5",
        "e5": "intfloat/e5-base-v2",
        "qwen2.5": "Qwen/Qwen2.5-7B",
        "qwen2.5-instruct": "Qwen/Qwen2.5-7B-Instruct",
    },
    "generator_model": args.model,
    "generator_batch_size": 1,
    "framework": "vllm",
    "gpu_id": "0, 1, 2, 3",
    "faiss_gpu": True,
    "retrieval_batch_size": 256,
    "gpu_memory_utilization": 0.5,
    "metrics": ["em", "f1", "acc", "recall", "precision"],
    "retrieval_topk": args.retrieval_top_k,
    "save_intermediate_data": True,
    "save_note": args.model + f"_iter{args.max_iter}",
}

answer_format = "answer"
max_iter = 10

config_dict = {**default_config, **config_dict}
config = Config(config_dict=config_dict)

dataset_path = config["dataset_path"]
split_path = os.path.join(dataset_path, "test.jsonl")
data_split = "test"
if not os.path.exists(split_path):
    if os.path.exists(os.path.join(dataset_path, "dev.jsonl")):
        data_split = "dev"
    elif os.path.exists(os.path.join(dataset_path, "val.jsonl")):
        data_split = "val"
    else:
        data_split = "None"

all_split = get_dataset(config)
test_data = all_split[data_split]

pipeline = ReasonRAGPipeline(config, prompt_template=None, answer_format=answer_format, max_iter=args.max_iter, max_children=2, max_rollouts=64)
output_dataset = pipeline.run(test_data, batch_size=1000, do_eval=True)
