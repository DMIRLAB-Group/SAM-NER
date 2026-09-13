import os
import tools
import uuid
from tqdm import tqdm


def taking_crossNER_raw_data(data_paths: list, save_dir: str):
    for path in tqdm(data_paths):
        data = tools.data_load(path)
        domain = path.split("/")[-1].split(".")[0]

        results = []
        for item in data:
            sentence = item.get("sentence")
            entities = item.get("entities")

            final_entities = []
            for info in entities:
                final_entities.append({
                    "name": info["name"],
                    "type": info["type"]
                })

            results.append({
                "id": str(uuid.uuid4()),
                "domain": domain,
                "sentence": sentence,
                "entities": final_entities
            })

        tools.data_save(results, os.path.join(save_dir, f"{domain}.json"))


def main():
    data_dir = ".."
    domains = ["ai", "literature", "music", "politics", "science"]
    data_paths = []
    for domain in domains:
        data_paths.append(os.path.join(data_dir, f"datasets/crossNER/raw/{domain}.json"))
    save_dir = os.path.join(data_dir, "datasets/crossNER/gold")
    taking_crossNER_raw_data(data_paths, save_dir)


if __name__ == "__main__":
    main()
