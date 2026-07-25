import os
import json
from typing import List, Dict, Any
from tqdm import tqdm


class ConvictionAgent:
    SYSTEM_PROMPT = "你是一个拥有丰富法律知识并且精通罪名预测的专家。"

    PROMPT_CHARGE = """你需要根据<犯罪事实>给出{name}在此案中涉及的罪名和法条。

<犯罪事实>
{case_fact}"""

    RES_FORMAT = "罪名: {accus}; 法条: {laws}"

    def __init__(self):
        pass

    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(tqdm(f.readlines(), desc=f"Loading {os.path.basename(file_path)}")):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)
                if item.get("id") is None:
                    item["id"] = idx
                data.append(item)
        return data

    def build_input(self, case_fact: str, name: str) -> str:
        return self.PROMPT_CHARGE.format(
            case_fact=case_fact,
            name=name
        )

    def build_output(self, accusations, laws) -> str:
        return self.RES_FORMAT.format(
            accus=accusations,
            laws=laws
        )

    def convert_sft_data(self, dataset: List[Dict[str, Any]]) -> List[Dict[str, Any]]:
        sft_data = []

        for sample_idx, sample in enumerate(tqdm(dataset, desc="Converting SFT data")):
            case_fact = sample["fact"]
            criminals_info = sample["criminals_info"]
            case_id = sample.get("id", sample_idx)

            for name, info in criminals_info.items():
                input_text = self.build_input(case_fact=case_fact, name=name)
                output_text = self.build_output(
                    accusations=info.get("accusations", []),
                    laws=info.get("laws", [])
                )

                piece = {
                    "instruction": self.SYSTEM_PROMPT,
                    "input": input_text,
                    "output": str(output_text),
                    "history": [],
                    "case_id": case_id,
                    "defendant": name
                }
                sft_data.append(piece)

        return sft_data

    @staticmethod
    def save_json(data: List[Dict[str, Any]], file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    @staticmethod
    def update_dataset_info(dataset_info_path: str, dataset_name: str, file_name: str):
        os.makedirs(os.path.dirname(dataset_info_path), exist_ok=True)

        if os.path.exists(dataset_info_path):
            with open(dataset_info_path, "r", encoding="utf-8") as f:
                dataset_info = json.load(f)
        else:
            dataset_info = {}

        dataset_info[dataset_name] = {
            "file_name": file_name,
            "columns": {
                "prompt": "instruction",
                "query": "input",
                "response": "output",
                "history": "history"
            }
        }

        with open(dataset_info_path, "w", encoding="utf-8") as f:
            json.dump(dataset_info, f, ensure_ascii=False, indent=4)

    def prepare_llamafactory_data(
        self,
        train_path: str,
        valid_path: str,
        test_path: str,
        output_dir: str,
        dataset_info_path: str
    ):
        train_data = self.load_jsonl(train_path)
        valid_data = self.load_jsonl(valid_path)
        test_data = self.load_jsonl(test_path)

        train_sft = self.convert_sft_data(train_data)
        valid_sft = self.convert_sft_data(valid_data)
        test_sft = self.convert_sft_data(test_data)

        train_file = os.path.join(output_dir, "hrn_train_sft.json")
        valid_file = os.path.join(output_dir, "hrn_valid_sft.json")
        test_file = os.path.join(output_dir, "hrn_test_sft.json")

        self.save_json(train_sft, train_file)
        self.save_json(valid_sft, valid_file)
        self.save_json(test_sft, test_file)

        self.update_dataset_info(dataset_info_path, "hrn_train_sft", "hrn_train_sft.json")
        self.update_dataset_info(dataset_info_path, "hrn_valid_sft", "hrn_valid_sft.json")
        self.update_dataset_info(dataset_info_path, "hrn_test_sft", "hrn_test_sft.json")

        print("Data prepared successfully.")
        print(f"Train samples: {len(train_sft)}")
        print(f"Valid samples: {len(valid_sft)}")
        print(f"Test samples: {len(test_sft)}")


if __name__ == "__main__":
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    train_path = os.path.join(project_root, "dataset/HRN/data_train_v5.03.jsonl")
    valid_path = os.path.join(project_root, "dataset/HRN/data_valid_v5.03.jsonl")
    test_path = os.path.join(project_root, "dataset/HRN/data_test_v5.03.jsonl")

    output_dir = os.path.join(project_root, "processed_data")
    dataset_info_path = os.path.join(project_root, "processed_data/dataset_info.json")

    agent = ConvictionAgent()
    agent.prepare_llamafactory_data(
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        output_dir=output_dir,
        dataset_info_path=dataset_info_path
    )
