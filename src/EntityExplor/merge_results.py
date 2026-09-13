import os
import json
from tqdm import tqdm
from src import tools


def add_tag_in_sentence(sentence, entities):
    start_tag = "<ENT>"
    end_tag = "</ENT>"

    # (start_index, end_index, text)
    matches = []
    for entity in entities:
        start = 0
        while True:
            # Find entity's position in sentence
            idx = sentence.find(entity, start)
            if idx == -1:
                break
            # [idx, idx + len)
            matches.append((idx, idx + len(entity), entity))
            start = idx + 1
    # Sorted(by length)
    matches.sort(key=lambda x: (x[0], -x[1]))

    # 3. Filter overlapping range
    final_intervals = []
    last_end = -1
    for start, end, text in matches:
        if start >= last_end:
            final_intervals.append((start, end))
            last_end = end

    result = []
    current_idx = 0

    for start, end in final_intervals:
        # not entity text
        result.append(sentence[current_idx:start])
        # entity text
        result.append(f"{start_tag}{sentence[start:end]}{end_tag}")
        current_idx = end

    # remaining text
    result.append(sentence[current_idx:])

    return "".join(result)


def CCR(ED_output, EE_output):
    denoise = {}
    ED_output_json = json.loads(ED_output)
    EE_output_json = json.loads(EE_output)

    ED_entities = []
    for _, values in ED_output_json.items():
        if values:
            ED_entities.extend(values)

    for key, values in EE_output_json.items():
        denoise[key] = []
        if values:
            for value in values:
                word_len = len(value.split())

                if word_len == 1:
                    if value not in ED_entities:
                        continue
                    else:
                        denoise[key].append(value)
                else:
                    denoise[key].append(value)

    return denoise


def get_final_result(ED_paths: list, EE_paths: list, save_dir: str):
    for ED_path, EE_path in zip(ED_paths, EE_paths):
        ED_result = tools.data_load(ED_path)
        EE_result = tools.data_load(EE_path)

        domain = ED_path.split("/")[-1].split(".")[0]

        id2output = {}
        for item in tqdm(ED_result):
            id_ = item.get("id")
            output = item.get("output")  # json string

            id2output[id_] = output

        results = []
        for item in tqdm(EE_result):
            id_ = item.get("id")
            sentence = item.get("sentence")
            output = item.get("output")  # json string

            ED_output = id2output[id_]

            denoise_output = CCR(ED_output, output)

            entities = set()
            for _, values in json.loads(ED_output).items():
                entities.update(values)

            for _, values in denoise_output.items():
                entities.update(values)

            tag_sentence = add_tag_in_sentence(sentence, entities)

            results.append({
                "id": id_,
                "sentence": sentence,
                "output": tag_sentence
            })

        tools.data_save(results, os.path.join(save_dir, f"{domain}.json"))


def main():
    data_dir = "../.."
    domains = ["ai", "literature", "music", "politics", "science"]
    EE_result_paths = []
    ED_result_paths = []
    for domain_ in domains:
        EE_result_paths.append(
            os.path.join(data_dir, f"data/crossNER/module2_EE/Llama3-8B/predict/{domain_}.json"))
        ED_result_paths.append(
            os.path.join(data_dir, f"data/crossNER/module1_ED/Llama3-8B/predict/{domain_}.json"))
    save_dir = os.path.join(data_dir, "data/crossNER/module2_EE/Llama3-8B/predict_merge_tag")
    get_final_result(ED_result_paths, EE_result_paths, save_dir)


if __name__ == "__main__":
    main()


