import json
import os
import re
from tqdm import tqdm
import uuid
from src import tools


def split_IEPile_NER_data(data_path, save_path):
    data = tools.data_load(data_path)

    result = []
    for item in tqdm(data, desc="spilt IEPile NER data"):
        task = item.get("item")
        source = item.get("source")
        instruction_str = item.get("instruction")
        output = item.get("output")

        if task == "NER":
            instruction_json = json.loads(instruction_str)
            sentence = instruction_json["input"]
            schema = instruction_json["schema"]

            result.append({
                "id": str(uuid.uuid4()),
                "source": source,
                "sentence": sentence,
                "schema": schema,
                "output": output
            })
        else:
            continue

    tools.data_save(result, save_path)


def remove_none_number_output(data_path: str, save_path: str):
    raw_data = tools.data_load(data_path)

    ec_data = []

    number = [
        "cardinal", "ordinal", "percent", "money", "Price", "quantity",
        "average ratings", "Rating", "rating", "date", "time", "year", "Hours"
    ]

    for item in tqdm(raw_data, desc="taking IEPile NER(raw) data..."):
        source = item.get("source")
        id_ = item.get("id")
        sentence = item.get("sentence")
        schema = item.get("schema")
        output = item.get("output")

        entities = {key: value for key, value in output.items() if value}

        if entities:
            for key in entities.keys():
                if key in number:
                    break
                else:
                    ec_data.append({
                        "id": id_,
                        "source": source,
                        "sentence": sentence,
                        "schema": schema,
                        "output": entities
                    })
                    break

    print(f"number of EC data:{len(ec_data)}")
    tools.data_save(ec_data, save_path)


def abstract_schema_mapping(data_path, schema_path, save_path):
    data = tools.data_load(data_path)
    schema_pre = tools.data_load(schema_path)

    invert_schema = {}
    for key, value in schema_pre.items():
        for v in value:
            invert_schema[v] = key

    results = []

    number = [
        "cardinal", "ordinal", "percent", "money", "Price", "quantity",
        "average ratings", "Rating", "rating", "date", "time", "year", "Hours"
    ]
    for item in tqdm(data, desc="schema mapping"):
        id_ = item.get("id")
        source = item.get("source")
        sentence = item.get("sentence")
        schema = item.get("schema")
        output = item.get("output")

        schema_new = set()
        for s in schema:
            if s in number:
                continue
            else:
                schema_new.add(invert_schema[s])

        output_new = {}
        if output:
            for key, value in output.items():
                if key in number:
                    continue
                else:
                    output_new[invert_schema[key]] = value
        else:
            output_new = output

        results.append({
            "id": id_,
            "source": source,
            "sentence": sentence,
            "schema": list(schema_new),
            "output": output_new
        })

    tools.data_save(results, save_path)


def add_instruction(data_path, save_path):
    ec_data = tools.data_load(data_path)

    results = []
    for item in tqdm(ec_data, desc="EC train add instruction"):
        id_ = item.get("id")
        source = item.get("source")
        sentence_org = item.get("sentence")
        schema = item.get("schema")
        output = item.get("output")

        entities = []
        for _, values in output.items():
            if values:
                entities.extend(values)

        sentence = add_tag_in_sentence(sentence_org, entities)

        if tools.is_chinese(sentence):
            instruction = f"你是一位实体分类专家。你的任务是：没有遗漏地对每一个 <ENT></ENT> 标记的实体进行分类，并根据句子语义严格从给定的 schema 中选择一个最恰当且唯一的实体类型。句子：{sentence} schema：{schema}。"
        else:
            instruction = f"You are an expert in entity classification. Your task is to classify each entity marked with <ENT></ENT> without omission and strictly select the most appropriate and unique entity type from the given schema based on the sentence semantics.sentence:{sentence} schema:{schema}"

        entities = []
        for key, values in output.items():
            for value in values:
                entities.append({
                    value: key
                })

        output_str = json.dumps(entities, ensure_ascii=False)
        results.append({
            "id": id_,
            "source": source,
            "instruction": instruction,
            "output": output_str
        })

    tools.data_save(results, save_path)


def add_tag_in_sentence(sentence: str, entities: list) -> str:
    tag_start = "<ENT>"
    tag_end = "</ENT>"

    if not isinstance(entities, list):
        print(f"{sentence} not a list")

    entities = sorted(entities, key=len, reverse=True)

    spans = []
    word_char_pattern = re.compile(r'\w')

    for entity in entities:
        if not entity.strip():
            continue

        if tools.is_chinese(entity):
            pattern = re.escape(entity)
        else:
            prefix = ''
            suffix = ''
            if word_char_pattern.match(entity[0]):
                prefix = r'(?<!\w)'
            if word_char_pattern.match(entity[-1]):
                suffix = r'(?!\w)'
            pattern = prefix + re.escape(entity) + suffix

        match = re.search(pattern, sentence)
        if match:
            spans.append(match.span())

    spans = sorted(spans, key=lambda x: x[0])
    merged = []
    for start, end in spans:
        if merged and start < merged[-1][1]:
            continue
        merged.append((start, end))

    result = list(sentence)
    for start, end in reversed(merged):
        result.insert(end, tag_end)
        result.insert(start, tag_start)

    return ''.join(result)


def main():
    data_dir = "../.."

    # step 1: Split NER data in IEPile raw data
    IEPile_raw_path = os.path.join(data_dir, "datasets/IEPile/IEPile.json")
    train_raw_save_path = os.path.join(data_dir, "data/IEPile/module3_EC/processed/train_raw.json")
    split_IEPile_NER_data(IEPile_raw_path, train_raw_save_path)

    # step2: Filter None&Number Output
    init_data_save_path = os.path.join(data_dir, "data/IEPile/module3_EC/processed/train_filter.json")
    remove_none_number_output(train_raw_save_path, init_data_save_path)

    # step3: Abstract Schema Mapping
    schema_mapping_path = os.path.join(data_dir, "data/IEPile/schema/schema_mapping.json")
    mapped_save_path = os.path.join(data_dir, "data/IEPile/module3_EC/processed/train_mapping.json")
    abstract_schema_mapping(init_data_save_path, schema_mapping_path, mapped_save_path)

    # step4: Add Instruction
    ec_train_save_path = os.path.join(data_dir, "data/IEPile/module3_EC/train.json")
    add_instruction(mapped_save_path, ec_train_save_path)


if __name__ == "__main__":
    main()
