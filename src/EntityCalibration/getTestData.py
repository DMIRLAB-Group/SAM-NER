import json
import os
import re
from tqdm import tqdm
from src import tools


def create_calibration_data(correction_data_path: list, IEPile_schema_define_path: str,
                           crossNER_schema_define_path: str, save_path: str):
    schema_IEPile_define = tools.data_load(IEPile_schema_define_path)
    schema_crossNER_define = tools.data_load(crossNER_schema_define_path)

    for path in correction_data_path:
        data = tools.data_load(path)
        domain_name = path.split("/")[-1].split(".")[0]

        final_data = []

        for item in tqdm(data, desc="create ECT test"):  # id sentence output
            start = "<ENT>"
            end = "</ENT>"

            id_ = item.get("id")
            sentence = item.get("sentence")
            clearing_sentence = re.sub(start, "", sentence)
            cleared_sentence = re.sub(end, "", clearing_sentence)
            output_pre = item.get("output")
            try:
                output2dict = json.loads(output_pre)
            except json.JSONDecodeError as e:
                print(f"error:{output_pre}, {e}")

            IEPile_schema_define = ""
            had_schema = set()
            if output2dict:
                for info in output2dict:
                    for key, value in info.items():
                        if value not in had_schema:
                            had_schema.add(value)
                            IEPile_schema_define += f"{value}:{schema_IEPile_define[value]}."

            # target schema definition
            domain_defines = schema_crossNER_define[domain_name]
            crossNER_schema = []
            crossNER_schema_define = ""
            for key, value in domain_defines.items():
                crossNER_schema.append(key)
                crossNER_schema_define += f"{key}:{value}."

            instruction = f"""
                You are a highly proficient expert in named entity recognition in semantic type mapping. Based on the given information, you need to strictly select the appropriate target domain type from the "Candidate Types" for each named entity of the "Initial Output".

                # Information
                Text: {cleared_sentence}
                Initial Output: {output_pre}
                Candidate Types: {crossNER_schema}
                Source Domain Type Definitions: {IEPile_schema_define}
                Target Domain Type Definitions: {crossNER_schema_define}
                # Rules
                1. First, you should review the `Text`, `Source Domain Type Definitions`, and `Target Domain Type Definitions`, and then select the target type from the candidate types based on this information.
                1. If an entity could belong to multiple target types, choose the most specific one. For example:
                   - If the entity could be both "researcher" and "person", choose "researcher";
                   - If it could be both "country" and "location", choose "country";
                   - If the entity could be both "organisation" and "university", choose "university".
                3. The entity names in your answer must exist in the `Initial Output`.
                2. The entity types in your answer must be selected from the `Candidate Types`.
                4. You need to give your final correction result after `final answer:` and add `@@` before and after the final correction result.
                5. Your answer is list and does not require any explanations or clarifications.
                # Check
                Check whether all entity types belong to the types in the given candidate list. If there are any situations that do not fall under this category, corrections shall be made in accordance with Rule 1.
                # output format
                final answer:
                @@
                [{{"entity name": "corrected type"}}, {{"entity name": "corrected type"}}, ...]
                @@
            """

            final_data.append({
                "id": id_,
                "instruction": instruction
            })

        tools.data_save(final_data, os.path.join(save_path, f"{domain_name}.json"))


def main():
    data_dir = "../.."

    domains = ["ai", "literature", "music", "politics", "science"]
    predict_data_path = []
    for domain_ in domains:
        predict_data_path.append(os.path.join(data_dir, f"data/crossNER/module3_EC/Llama3-8B/predict/{domain_}.json"))
    IEPile_schema_define_path = os.path.join(data_dir, "data/crossNER/schema/schema_define_IEPile.json")
    crossNER_schema_define_path = os.path.join(data_dir, "data/crossNER/schema/schema_define_crossNER.json")
    save_dir = os.path.join(data_dir, "data/crossNER/module4_ETC/calibration")

    create_calibration_data(predict_data_path, IEPile_schema_define_path, crossNER_schema_define_path, save_dir)


if __name__ == "__main__":
    main()


