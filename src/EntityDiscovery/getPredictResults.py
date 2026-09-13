import os
import sys
import re
import json
from tqdm import tqdm
from src import tools


def get_predict_data_add_id(data_path, id_data_path, save_path):
    pattern = re.compile(r'(\{.*?\})(?=<\|eot_id\|>)', re.DOTALL)
    predict_pattern = r'(\{[\s\S]*?\})'
    predict_data = tools.data_load_jsonl(data_path)
    id_data = tools.data_load(id_data_path)

    id_dict = {}
    for item in tqdm(id_data, desc="mapping sentence:id"):
        id_ = item.get("id")
        sentence = item.get("sentence")
        id_dict[tools.remove_symbol(sentence)] = id_

    results = []
    for item in tqdm(predict_data, desc="get predict result"):
        instruction_str = pattern.search(item.get("prompt")).group(1)
        predict_str = re.search(predict_pattern, item.get("predict")).group(1)

        instruction = json.loads(instruction_str)

        sentence = instruction.get("input")
        id_ = id_dict[tools.remove_symbol(sentence)]

        results.append({
            "id": id_,
            "sentence": sentence,
            "output": predict_str
        })
    tools.data_save(results, save_path)


def main(domain):
    data_dir = "../.."
    predict_data_path = os.path.join(data_dir, "data/crossNER/module1_ED/Llama3-8B/generated_predictions.jsonl")
    id_data_path = os.path.join(data_dir, f"datasets/crossNER/gold/{domain}.json")
    save_path = os.path.join(data_dir, f"data/crossNER/module1_ED/Llama3-8B/predict/{domain}.json")
    get_predict_data_add_id(predict_data_path, id_data_path, save_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        domain_arg = sys.argv[1]
        print(f"Start processing domain: {domain_arg}")
        main(domain_arg)
        print(f"Successfully processed {domain_arg}")
    else:
        print("Error: Please provide a domain argument")
        sys.exit(1)

