import argparse
import json
import logging
import os
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List

from datasets import Dataset, load_dataset
from tqdm import tqdm

logging.basicConfig(
    level=logging.INFO, format="%(asctime)s - %(levelname)s - %(message)s"
)


class OllamaTranslator:
    def __init__(self, url: str, model: str, assistant_prompt: str):
        self.url = url
        self.model = model
        self.assistant_prompt = assistant_prompt

    def query_model(self, prompt: str) -> str:
        data = {
            "model": self.model,
            "seed": 676,
            "temperature": 1.0,
            "top_p": 1,
            "messages": [
                {"role": "assistant", "content": self.assistant_prompt},
                {"role": "user", "content": prompt},
            ],
        }
        payload = json.dumps(data).encode("utf-8")

        try:
            request = urllib.request.Request(self.url, data=payload, method="POST")
            request.add_header("Content-Type", "application/json")

            with urllib.request.urlopen(request) as response:
                response_data = ""
                for line in response:
                    response_json = json.loads(line.decode("utf-8"))
                    response_data += response_json["message"]["content"]
                return response_data
        except urllib.error.URLError as e:
            logging.error(f"Error querying Ollama model: {e}")
            return ""


class DatasetTranslator:
    def __init__(
        self, translator: OllamaTranslator, dataset: Dataset, conversation_column: str
    ):
        self.translator = translator
        self.dataset = dataset
        self.conversation_column = conversation_column

    def translate_conversation(
        self, conversation: List[Dict[str, str]]
    ) -> List[Dict[str, str]]:
        translated_conversation = []
        for message in conversation:
            translated_message = message.copy()
            if message["from"] == "gpt":
                translated_message["value"] = self.translator.query_model(
                    message["value"]
                )
            translated_conversation.append(translated_message)
        return translated_conversation

    def process_dataset(self, num_workers: int = 4) -> List[List[Dict[str, str]]]:
        translated_data = []
        with ThreadPoolExecutor(max_workers=num_workers) as executor:
            future_to_item = {
                executor.submit(
                    self.translate_conversation,
                    json.loads(item[self.conversation_column]),
                ): item
                for item in self.dataset
            }
            for future in tqdm(
                as_completed(future_to_item),
                total=len(self.dataset),
                desc="Translating dataset",
            ):
                translated_data.append(future.result())
        return translated_data


def save_translated_dataset(translated_data: List[Any], output_file: str):
    with open(output_file, "w", encoding="utf-8") as f:
        for item in translated_data:
            f.write(json.dumps(item, ensure_ascii=False) + "\n")


def main():
    parser = argparse.ArgumentParser(
        description="Translate HuggingFace dataset conversations using Ollama"
    )
    parser.add_argument(
        "--ds_id", type=str, default="liuhaotian/LLaVA-Instruct-150K", help="dataset id"
    )
    parser.add_argument("--split", type=str, default="train", help="dataset split")
    parser.add_argument(
        "--num_samples", type=int, default=None, help="number of samples to load"
    )
    parser.add_argument(
        "--save_dir", type=str, default="data", help="directory to save data"
    )
    parser.add_argument(
        "--ollama_url",
        type=str,
        default="http://localhost:11434/api/chat",
        help="Ollama API URL",
    )
    parser.add_argument(
        "--ollama_model", type=str, default="qwen2.5:7b", help="Ollama model to use"
    )
    parser.add_argument(
        "--conversation_column",
        type=str,
        default="conversations",
        help="name of the conversation column in the dataset",
    )
    parser.add_argument(
        "--num_workers",
        type=int,
        default=4,
        help="number of worker threads for translation",
    )
    args = parser.parse_args()

    try:
        ds = load_dataset(args.ds_id)[args.split]
        if args.num_samples is not None:
            ds = ds.select(range(args.num_samples))

        logging.info(f"Loaded {len(ds)} samples from dataset {args.ds_id}")

        assistant_prompt = "Translate to Spanish the following text, avoiding special tokens and code snippets"
        translator = OllamaTranslator(
            args.ollama_url, args.ollama_model, assistant_prompt
        )
        logging.info(f"Using Ollama model {args.ollama_model} at {args.ollama_url}")

        dataset_translator = DatasetTranslator(translator, ds, args.conversation_column)

        logging.info("Translating dataset...")

        translated_data = dataset_translator.process_dataset(
            num_workers=args.num_workers
        )

        os.makedirs(args.save_dir, exist_ok=True)
        output_file = os.path.join(args.save_dir, f"translated_{args.split}.jsonl")
        save_translated_dataset(translated_data, output_file)

        logging.info(f"Translated dataset saved to {output_file}")

    except Exception as e:
        logging.error(f"An error occurred: {e}")


if __name__ == "__main__":
    main()
