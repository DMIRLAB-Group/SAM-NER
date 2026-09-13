import json
import os
import re
from tqdm import tqdm
import random
from src import tools


def take_pile_ner4train(data_path, save_path):
    data = tools.data_load(data_path)

    schema_pattern = re.compile(r"What describes (.*?) in the text?")
    results = []

    for item in tqdm(data, desc="get explorer train data"):
        infos = item.get("conversations")
        sentence = infos[0]["value"]
        sentence = re.sub(r"^Text:\s*Q:\s*\n\n?", "", sentence, flags=re.IGNORECASE)
        sentence = re.sub(r"^Text:\s*", "", sentence, flags=re.IGNORECASE).strip()

        schema = []
        output = {}

        for i in range(len(infos) - 1):
            msg = infos[i]

            if msg["from"] == "human":
                match = schema_pattern.search(msg["value"])
                if match:
                    type_ = match.group(1).strip()
                    schema.append(type_)

                    gpt_msg = infos[i + 1]["value"]

                    try:
                        parsed = json.loads(gpt_msg)
                    except Exception as e:
                        parsed = []

                    output[type_] = parsed

        random.shuffle(schema)
        results.append({
            "sentence": sentence,
            "schema": schema,
            "output": output
        })

    tools.data_save(results, save_path)
    print(f"data size: {len(results)}")


def EE_trainData_add_instruction(data_path, save_path):
    data = tools.data_load(data_path)
    instruction_base = "You are a helpful assistant for extracting named entities. Given a passage and a schema, your task is to extract all named entities that conform to the schema type."

    results = []
    for item in tqdm(data, desc="add instruction"):
        sentence = item.get("sentence")
        schema = item.get("schema")
        output = json.dumps(item.get("output"), ensure_ascii=False)

        instruction = instruction_base + f"Schema:{schema}. Passage: {sentence}"

        results.append({
            "sentence": sentence,
            "instruction": instruction,
            "output": output
        })

    tools.data_save(results, save_path)
    print(f"data size: {len(results)}")


def main():
    data_dir = "../.."
    pile_data_path = os.path.join(data_dir, f"datasets/PileNER/Pile-NER-type.json")
    pile_format_save_path = os.path.join(data_dir, f"data/PileNER/Pile_NER.json")

    # step1: Get raw train data
    take_pile_ner4train(pile_data_path, pile_format_save_path)

    # step2: Add instruction
    pile_train_save_path = os.path.join(data_dir, f"data/PileNER/module2_EE/train.json")
    EE_trainData_add_instruction(pile_format_save_path, pile_train_save_path)


if __name__ == "__main__":
    main()

