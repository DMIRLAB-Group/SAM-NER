import json
import os
import re
from tqdm import tqdm
from src import tools


def get_ETC_result(data_path, id_data_path, save_path):
    sentence_pattern = re.compile(r"Text:\s*(.*?)\s*Initial Output:", re.DOTALL)
    predict_pattern_1 = re.compile(
        r"(?:Final Answer:|final answer:)\s*@{2,}\s*(.*?)(?=\s*@{2,}|$)",
        re.DOTALL
    )
    predict_pattern_2 = re.compile(
        r"@{2,}\s*(.*?)(?=\s*@{2,}|$)",
        re.DOTALL
    )

    predict_data = tools.data_load_jsonl(data_path)
    id_data = tools.data_load(id_data_path)

    sentence2id = {}
    for item in tqdm(id_data, desc="sentence:id"):
        instruction = item.get("instruction")
        sentence = sentence_pattern.search(instruction).group(1)
        sentence2id[tools.remove_symbol(sentence)] = item.get("id")

    results = []
    for item in tqdm(predict_data, desc="get predict result"):
        prompt = item.get("prompt")
        predict = item.get("predict")

        sentence = sentence_pattern.search(prompt).group(1)
        id_ = sentence2id[tools.remove_symbol(sentence)]
        output_match = predict_pattern_1.search(predict)
        if not output_match:
            output_match = predict_pattern_2.search(predict)

        output_json = []
        if output_match:
            start = "<ENT>"
            end = "</ENT>"
            try:
                output = json.loads(output_match.group(1))
                for ents in output:
                    for name, type_ in ents.items():
                        name = re.sub(start, "", name)
                        name = re.sub(end, "", name)
                        output_json.append({
                            name: type_
                        })

                output = output_json
            except Exception as e:
                output = output_match.group(1)

        results.append({
            "id": id_,
            "sentence": sentence,
            "output": output
        })
    tools.data_save(results, save_path)


def main(domain):
    data_dir = "../.."

    predict_data_path = os.path.join(data_dir, "data/crossNER/module4_ETC/Llama3-8B/generated_predictions.jsonl")
    id_data_path = os.path.join(data_dir, f"data/crossNER/module4_ETC/calibration/{domain}.json")
    save_path = os.path.join(data_dir, f"data/crossNER/module4_ETC/Llama3-8B/predict/{domain}.json")
    get_ETC_result(predict_data_path, id_data_path, save_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        domain_arg = sys.argv[1]
        print(f"Start processing domain: {domain_arg}")
        main(domain_arg)
        print(f"Successfully processed {domain_arg}")
    else:
        print("Error: Please provide a domain argument")
        sys.exit(1)


