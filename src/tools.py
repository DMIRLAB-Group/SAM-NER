import json
import re
from tqdm import tqdm


# read JSON
def data_load(data_path):
    try:
        with open(data_path, 'r', encoding="utf-8") as f:
            data = json.load(f)
        print(f"Loading '{data_path}' completed")
        return data
    except json.JSONDecodeError as e:
        print(f"'{data_path}' decoding failed！！！")
    except Exception as e:
        print(f"Loading '{data_path}' failed！！！")


# read JSON string
def data_load_string(sentence):
    try:
        return json.loads(sentence)
    except json.JSONDecodeError as e:
        print(f"Not Json string！！！")
    except Exception as e:
        print(f"Loading Failed！！！")


# read jsonl
def data_load_jsonl(data_path):
    all_data = []
    with open(data_path, "r", encoding="utf-8") as f:
        for line in f:
            try:
                data = json.loads(line)
                all_data.append(data)
            except json.JSONDecodeError:
                print(f"Failed to read the current row of data:{line.strip()}")

    return all_data


# save to JSON
def data_save(data, save_path):
    try:
        with open(save_path, 'w', encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)
        print(f"save '{save_path}' successfully")
    except Exception as e:
        print(f"save '{save_path}' failed！reason:{e}")


def is_chinese(sentence: str) -> bool:
    for char in sentence:
        if '\u4e00' <= char <= '\u9fff':
            return True
    return False


def remove_symbol(sentence):
    return re.sub(r'\s+', '', sentence).lower()


def split_EE_predict_data(data_path):
    data = data_load_jsonl(data_path)
    pattern = re.compile(r"sentence:\s*(.*?)\s*<\|eot_id\|>", re.DOTALL)

    final_data = []

    for item in tqdm(data, desc="split EE predict data"):
        prompt = item.get("prompt")
        output = item.get("predict")
        match = pattern.search(prompt)
        if match:
            sentence = match.group(1).strip()
            final_data.append({
                "sentence": sentence,
                "output": output
            })
        else:
            print(f"match error:{output}")

    return final_data


def split_EC_predict_data(data_path):
    data = data_load_jsonl(data_path)
    pattern = re.compile(r"sentence:\s*(.*?)\s*schema:", re.DOTALL)

    final_data = []

    for item in tqdm(data, desc="split EC predict data"):
        prompt = item.get("prompt")
        output = item.get("predict")
        match = pattern.search(prompt)
        if match:
            sentence = match.group(1).strip()
            final_data.append({
                "sentence": sentence,
                "output": output
            })
        else:
            print(f"match error:{output}")

    return final_data


def get_tag_content(sentence):
    start = "<ENT>"
    end = "</ENT>"
    pattern = re.compile(r"<ENT>\s*(.*?)\s*</ENT>", re.DOTALL)

    entities = set(re.findall(pattern, sentence))

    return list(entities)
