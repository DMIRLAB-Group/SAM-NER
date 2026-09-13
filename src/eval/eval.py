import ast
import os
from tqdm import tqdm
from src import tools


def cal_f1(gold_path: list, predict_path: list, save_path: str) -> None:
    domain_gold = {}
    for path in gold_path:
        domain = os.path.basename(path).split(".")[0]

        data = tools.data_load(path)

        domain_gold[domain] = {}
        for item in data:
            id_ = item.get("id")
            entities = item.get("entities", [])

            domain_gold[domain][id_] = [
                {"name": e["name"].strip(), "type": e["type"].strip()}
                for e in entities
            ]

    all_results = []
    total_tp = total_fp = total_fn = 0
    avg_f1 = 0
    for path in tqdm(predict_path, desc="计算F1"):
        domain = os.path.basename(path).split(".")[0]

        predict_data = tools.data_load(path)
        gold_data = domain_gold.get(domain, {})

        tp, fp, fn = 0, 0, 0

        predict_dict = {}
        for item in predict_data:
            id_ = item.get("id")
            output = item.get("output", [])  # [{"A": "type1"}, {"B": "type2"}]
            entities = []
            if isinstance(output, list):
                for ent in output:
                    if isinstance(ent, dict):
                        for k, v in ent.items():
                            if isinstance(k, list) or isinstance(v, list):
                                print(f"error of key or value is list: {id_}")
                            try:
                                entities.append({"name": k.strip(), "type": v.strip()})
                            except Exception:
                                print(f"error of key or value: {id_}")
                                entities.append({"name": k, "type": v})
                    else:
                        print("entity type not dict")
            elif isinstance(output, str):
                print(f"string data: {id_}")
                try:
                    output = ast.literal_eval(output)
                except Exception as e:
                    print(f"{domain} error: {id_}")
                for ent in output:
                    if isinstance(ent, dict):
                        for k, v in ent.items():
                            if isinstance(k, list) or isinstance(v, list):
                                print(f"error of key or value is list: {id_}")
                            try:
                                entities.append({"name": k.strip(), "type": v.strip()})
                            except Exception:
                                print(f"error of key or value: {id_}")
                                entities.append({"name": k, "type": v})
                    else:
                        print(f"{id_} entity type not dict")
            predict_dict[id_] = entities

        gold_ids = set(gold_data.keys())
        pred_ids = set(predict_dict.keys())
        all_ids = gold_ids | pred_ids

        for id_ in tqdm(all_ids, desc=f"calculating {domain} score"):
            gold_entities = gold_data.get(id_, [])
            pred_entities = predict_dict.get(id_, [])

            gold_set = {(e["name"], e["type"]) for e in gold_entities}
            pred_set = {(e["name"], e["type"]) for e in pred_entities}

            tp += len(gold_set & pred_set)
            fp += len(pred_set - gold_set)
            fn += len(gold_set - pred_set)

        precision = tp / (tp + fp) if (tp + fp) else 0.0
        recall = tp / (tp + fn) if (tp + fn) else 0.0
        f1 = 2 * precision * recall / (precision + recall) if (precision + recall) else 0.0

        result = {
            "domain": domain,
            "Precision": round(precision, 4),
            "Recall": round(recall, 4),
            "F1": round(f1, 4),
        }
        avg_f1 += f1
        all_results.append(result)

        total_tp += tp
        total_fp += fp
        total_fn += fn

    all_results.append({
        "domain": "micro_avg",
        "F1": round(avg_f1 / 5.0, 4)
    })

    tools.data_save(all_results, save_path)


if __name__ == "__main__":
    data_dir = "../.."

    domains = ["ai", "literature", "music", "politics", "science"]
    predict_path = []
    gold_path = []
    for domain in domains:
        gold_path.append(os.path.join(data_dir, f"datasets/crossNER/gold/{domain}.json"))
        predict_path.append(os.path.join(data_dir, f"data/crossNER/module4_ETC/Llama3-8B/predict/{domain}.json"))
    save_path = os.path.join(data_dir, "data/eval/Llama3-8B/F1_score.json")
