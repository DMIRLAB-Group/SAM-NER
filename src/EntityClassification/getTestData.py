import os
from tqdm import tqdm
from src import tools


def create_test_data(predict_path: list, save_dir: str):
    schema = ["person", "organization", "location", "biology", "medicine", "food", "vehicle", "creative_work", "event", "artifact", "computer_science", "political", "science", "misc"]

    for path in predict_path:
        test_data = []
        predict_data = tools.data_load(path)
        domain = path.split("/")[-1].split(".")[0]

        for item in tqdm(predict_data):
            instruction = f"You are an expert in entity classification. Your task is to classify each entity marked with <ENT></ENT> without omission and strictly select the most appropriate and unique entity type from the given schema based on the sentence semantics."
            id_ = item.get("id")
            sentence = item.get("output")
            instruction = instruction + f"sentence:{sentence} schema:{schema}"
            test_data.append({
                "id": id_,
                "sentence": sentence,
                "instruction": instruction
            })

        tools.data_save(test_data, os.path.join(save_dir, f"{domain}.json"))


def main():
    data_dir = "../.."

    domains = ["ai", "literature", "music", "politics", "science"]
    predict_path = []
    for domain_ in domains:
        predict_path.append(os.path.join(data_dir, f"data/crossNER/module2_EE/Llama3-8B/predict_merge_tag/{domain_}.json"))
    save_dir = os.path.join(data_dir, "data/crossNER/module3_EC/test")
    create_test_data(predict_path, save_dir)


if __name__ == "__main__":
    main()

