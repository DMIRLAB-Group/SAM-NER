import os
import sys
import re
from tqdm import tqdm
from src import tools


def get_EE_result(predict_data_path, id_data_path, save_path):
    data = tools.data_load_jsonl(predict_data_path)
    pattern = re.compile(r"Passage:\s*(.*?)\s*<\|eot_id\|>", re.DOTALL)

    predict_data = []

    for item in tqdm(data, desc="get EE result"):
        prompt = item.get("prompt")
        output = item.get("predict")
        match = pattern.search(prompt)
        if match:
            sentence = match.group(1).strip()
            predict_data.append({
                "sentence": sentence,
                "output": output
            })
        else:
            print(f"re match error:{output}")
    id_data = tools.data_load(id_data_path)

    sentence2id = {}
    for item in tqdm(id_data, desc="sentence2id"):
        id_ = item.get("id")
        sentence = tools.remove_symbol(item.get("sentence"))
        sentence2id[sentence] = id_

    final = []
    for item in tqdm(predict_data, desc="predict result match id"):
        output = item.get("output")
        sentence = item.get("sentence")
        id_ = sentence2id[tools.remove_symbol(sentence)]
        final.append({
            "id": id_,
            "sentence": sentence,
            "output": output
        })
    tools.data_save(final, save_path)


def main(domain):
    data_dir = "../.."
    predict_data_path = os.path.join(data_dir, "data/crossNER/module2_EE/Llama3-8B/generated_predictions.jsonl")
    id_data_path = os.path.join(data_dir, f"datasets/crossNER/gold/{domain}.json")
    save_path = os.path.join(data_dir, f"data/crossNER/module2_EE/Llama3-8B/predict/{domain}.json")
    get_EE_result(predict_data_path, id_data_path, save_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        domain_arg = sys.argv[1]
        print(f"Start processing domain: {domain_arg}")
        main(domain_arg)
        print(f"Successfully processed {domain_arg}")
    else:
        print("Error: Please provide a domain argument")
        sys.exit(1)

