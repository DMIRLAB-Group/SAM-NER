import os
import sys
import json
import re
from tqdm import tqdm
from src import tools


def supplement_forget_tag_entity(sentence: str, output: list) -> list:
    entity_pattern = re.compile(r"<ENT>(.*?)</ENT>")
    sentence_tag = entity_pattern.findall(sentence)
    entities = []
    for cell in output:
        for entity, type_ in cell.items():
            entities.append(entity)

    sentence_tag_set = set(sentence_tag)
    entities_set = set(entities)
    result = sentence_tag_set - entities_set
    if result:
        return list(result)


def get_EC_result(predict_path, id_data_path, save_path):
    predict = tools.split_EC_predict_data(predict_path)
    id_data = tools.data_load(id_data_path)
    sentence2id = {}

    for item in tqdm(id_data, desc="sentence : id"):
        sentence = tools.remove_symbol(item.get("sentence"))
        sentence2id[sentence] = item.get("id")

    results = []
    for item in tqdm(predict, desc="get predict result"):
        sentence_pre = item.get("sentence")
        id_ = sentence2id[tools.remove_symbol(sentence_pre)]
        output_str = item.get("output")
        try:
            output = json.loads(output_str)
        except json.JSONDecodeError:
            print(f"DecodeError: {sentence_pre}")
        forget_entities = supplement_forget_tag_entity(sentence_pre, output)
        if forget_entities:
            for entity in forget_entities:
                output.append({entity: "misc"})
        final_output = json.dumps(output, ensure_ascii=False)
        results.append({
            "id": id_,
            "sentence": sentence_pre,
            "output": final_output
        })

    tools.data_save(results, save_path)
    print(f"number of predict:{len(results)}, number of id data:{len(id_data)}")


def main(domain):
    data_dir = "../.."

    predict_path = os.path.join(data_dir, "data/crossNER/module3_EC/Llama3-8B/generated_predictions.jsonl")
    id_data_path = os.path.join(data_dir, f"data/crossNER/module3_EC/test/{domain}.json")
    save_path = os.path.join(data_dir, f"data/crossNER/module3_EC/Llama3-8B/predict/{domain}.json")
    get_EC_result(predict_path, id_data_path, save_path)


if __name__ == "__main__":
    if len(sys.argv) > 1:
        domain_arg = sys.argv[1]
        print(f"Start processing domain: {domain_arg}")
        main(domain_arg)
        print(f"Successfully processed {domain_arg}")
    else:
        print("Error: Please provide a domain argument")
        sys.exit(1)
