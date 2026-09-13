import os
from tqdm import tqdm
from src import tools


def create_test_data(data_path: list, schema_path: str, save_dir: str) -> None:
    for path in data_path:
        data = tools.data_load(path)
        results = []
        domain_name = path.split("/")[-1].split(".")[0]

        schema = tools.data_load(schema_path)[domain_name]

        for item in tqdm(data, desc=f"{domain_name} add instruction"):
            instruction = "You are a helpful assistant for extracting named entities. Given a passage and a schema, your task is to extract all named entities that conform to the schema type."

            id_ = item.get("id")
            domain = item.get("domain")
            sentence = item.get("sentence")
            instruction = instruction + f"Schema:{schema}. Passage: {sentence}"
            results.append({
                "id": id_,
                "domain": domain,
                "sentence": sentence,
                "instruction": instruction
            })
        tools.data_save(results, os.path.join(save_dir, f"{domain_name}.json"))


def main():
    data_dir = "../.."
    domains = ["ai", "literature", "music", "politics", "science"]
    data_paths = []
    for domain in domains:
        data_paths.append(os.path.join(data_dir, f"datasets/crossNER/gold/{domain}.json"))
    schema_path = os.path.join(data_dir, "data/crossNER/schema/schema_crossner.json")
    test_save_dir = os.path.join(data_dir, "data/crossNER/module2_EE/test")
    create_test_data(data_paths, schema_path, test_save_dir)


if __name__ == "__main__":
    main()


