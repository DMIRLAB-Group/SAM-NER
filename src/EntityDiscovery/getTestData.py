import os
import json
from tqdm import tqdm
from src import tools


def create_test_data(data_paths, schema_path, save_dir):
    for path in tqdm(data_paths):
        data = tools.data_load(path)
        schemas = tools.data_load(schema_path)

        domain = path.split("/")[-1].split(".")[0]
        schema = schemas[domain]
        results = []

        for item in tqdm(data, desc="create instruction data"):
            id_ = item.get("id")
            sentence = item.get("sentence")

            instruct = "You are an expert in named entity recognition. Please extract entities that match the schema definition from the input. Return an empty list if the entity type does not exist. Please respond in the format of a JSON string."
            instruction = {
                "instruction": instruct,
                "schema": schema,
                "input": sentence
            }
            instruction = json.dumps(instruction, ensure_ascii=False)
            results.append({
                "id": id_,
                "instruction": instruction
            })

        tools.data_save(results, os.path.join(save_dir, f"{domain}.json"))


def main():
    data_dir = "../.."
    domains = ["ai", "literature", "music", "politics", "science"]
    data_paths = []
    for domain in domains:
        data_paths.append(os.path.join(data_dir, f"datasets/crossNER/gold/{domain}.json"))
    save_dir = os.path.join(data_dir, "data/crossNER/module1_ED/test")
    schema_path = os.path.join(data_dir, "data/crossNER/schema/schema_crossner.json")
    create_test_data(data_paths, schema_path, save_dir)


if __name__ == "__main__":
    main()
