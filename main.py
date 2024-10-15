import argparse
import json
import os
import urllib.request
from tqdm import tqdm

from datasets import load_dataset

parser = argparse.ArgumentParser()
parser.add_argument(
    "--ds_id", type=str, default="liuhaotian/LLaVA-Instruct-150K", help="dataset id"
)
parser.add_argument("--split", type=str, default="train", help="dataset split")
parser.add_argument(
    "--num_samples", type=int, default=None, help="number of samples to load"
)
parse.add_argument(
    "--save_dir", type=str, default="data", help="directory to save data"
)
args = parser.parse_args()

ds = load_dataset(args.ds_id)[args.split]
if args.num_samples is not None:
    ds = ds.select(range(args.num_samples))

# ollama default URL
URL = "http://localhost:11434/api/chat"
MODEL = "qwen2.5:7b"

assistant_prompt = (
    "Translate to Spanish the following text, avoiding special tokens and code snippets"
)

messages = [{"role": "assistant", "content": assistant_prompt}]


def query_model(prompt: str):
    data = {
        "model": MODEL,
        "seed": 676,
        "temperature": 1.0,
        "top_p": 1,
        "messages": [
            {"role": "assistant", "content": assistant_prompt},
            {"role": "user", "content": prompt},
        ],
    }
    # Convert the dictionary to a JSON formatted string and encode it to bytes
    payload = json.dumps(data).encode("utf-8")

    # Create a request object, setting the method to POST and adding necessary headers
    request = urllib.request.Request(url, data=payload, method="POST")
    request.add_header("Content-Type", "application/json")

    # Send the request and capture the response
    response_data = ""
    with urllib.request.urlopen(request) as response:
        # Read and decode the response
        while True:
            line = response.readline().decode("utf-8")
            if not line:
                break
            response_json = json.loads(line)
            response_data += response_json["message"]["content"]

    return response_data


