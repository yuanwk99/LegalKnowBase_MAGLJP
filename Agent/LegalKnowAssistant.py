import argparse
import ast
import copy
import json
import os
import re
import sys
from concurrent.futures import ThreadPoolExecutor, as_completed
from typing import Any, Dict, List, Optional, Tuple

from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from utils_drdz.call_llm import ask_tyqw
from utils_drdz.legal_knowledge_retrieval import LegalKnowledgeRetriever


class LegalKnowAssistant:
    SYSTEM_PROMPT = "你是一个拥有丰富法律知识并且精通量刑因果分析的司法专家。"

    PROMPT_GENERATE_GRAPH = """你需要根据[犯罪事实]、[法院解释]、[判决结果]和检索得到的法律知识，生成刑期因果图谱（Legal Event Logic Graph, LELG）。

图谱要求：
[1] 图谱是有向无环图，由 nodes 和 edges 两部分组成。
[2] 节点类型只有三类：原子事实节点 f、法律知识节点 k、刑期结果节点 p。
[3] 边只允许两类：原子事实节点指向法律知识节点；法律知识节点指向刑期结果节点。
[4] 原子事实节点必须来自犯罪事实或法院解释中影响量刑的具体事实，不要写笼统概括。
[5] 法律知识节点必须对应影响刑期的量刑规则，例如量刑幅度、基准刑、从重、从轻、减轻、累犯、坦白、自首、从犯、未遂等。
[6] 多名被告人的共同事实可以共用节点，但每名被告人应有独立的刑期结果节点。
[7] 输出必须是 Python 代码块，仅包含 nodes 和 edges 两个变量，不要输出解释。

输出格式：
```python
nodes = {
    "f1": {"type": "f", "explanation": "..."},
    "k1": {"type": "k", "explanation": "..."},
    "p1": {"type": "p", "explanation": "[被告A]最终被判处..."}
}

edges = [
    ("f1", "k1"),
    ("k1", "p1")
]
```

参考示例：
{demo}

轮到你了：
<input>
[犯罪事实]
{case_fact}

[法院解释]
{explanation}

[判决结果]
{sentence_result}

[通用法律理论]
{general_law}

[特定法律知识]
{specific_law}

<output>
"""

    PROMPT_GENERATE_GRAPH_DEMO = """```python
nodes = {
    "f1": {"type": "f", "explanation": "[被告A]、[被告B]结伙秘密窃取他人摩托车，价值人民币1578元"},
    "f2": {"type": "f", "explanation": "[被告A]在监视居住期间又犯新罪"},
    "f3": {"type": "f", "explanation": "[被告A]、[被告B]当庭自愿认罪，且被盗摩托车已发还被害人"},
    "k1": {"type": "k", "explanation": "盗窃公私财物数额较大，应在相应法定刑幅度内确定量刑起点"},
    "k2": {"type": "k", "explanation": "在监视居住期间再犯新罪，依法应当数罪并罚"},
    "k3": {"type": "k", "explanation": "当庭自愿认罪、退赃退赔，可以从轻处罚"},
    "p1": {"type": "p", "explanation": "[被告A]最终被判处有期徒刑一年"},
    "p2": {"type": "p", "explanation": "[被告B]最终被判处有期徒刑七个月"}
}

edges = [
    ("f1", "k1"),
    ("f2", "k2"),
    ("f3", "k3"),
    ("k1", "p1"),
    ("k2", "p1"),
    ("k3", "p1"),
    ("k1", "p2"),
    ("k3", "p2")
]
```"""

    def __init__(self, kb_path: Optional[str] = None, general_knowledge_path: Optional[str] = None):
        if kb_path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            kb_path = os.path.join(project_root, "LegalKnowBase", "legal_knowledge_base.json")
        if general_knowledge_path is None:
            project_root = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
            general_knowledge_path = os.path.join(
                project_root, "LegalKnowBase", "general_knowledge_other_charges.txt"
            )
        self.kb_path = kb_path
        self.general_knowledge_path = general_knowledge_path
        self.retriever = LegalKnowledgeRetriever(kb_path, general_knowledge_path)

    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
        data = []
        with open(file_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(tqdm(f.readlines(), desc=f"Loading {os.path.basename(file_path)}")):
                line = line.strip()
                if line:
                    item = json.loads(line)
                    if item.get("id") is None:
                        item["id"] = idx
                    data.append(item)
        return data

    @staticmethod
    def save_json(data: List[Dict[str, Any]], file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            json.dump(data, f, ensure_ascii=False, indent=4)

    @staticmethod
    def save_jsonl(data: List[Dict[str, Any]], file_path: str):
        os.makedirs(os.path.dirname(file_path), exist_ok=True)
        with open(file_path, "w", encoding="utf-8") as f:
            for item in data:
                f.write(json.dumps(item, ensure_ascii=False) + "\n")

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

    @staticmethod
    def build_sentence_result(case: Dict[str, Any], include_penalty: bool = True) -> str:
        lines = []
        defendants = case.get("defendants") or list(case.get("criminals_info", {}).keys())
        for name in defendants:
            info = case.get("criminals_info", {}).get(name, {})
            accusations = "，".join(str(x) for x in info.get("accusations", []))
            laws = "，".join(str(x) for x in info.get("laws", []))
            if include_penalty:
                term = info.get("term", "None")
                lines.append(f"{name}: 触犯刑法第{laws}条，构成{accusations}，判处{term}")
            else:
                lines.append(f"{name}: 触犯刑法第{laws}条，构成{accusations}，刑期=None")
        return "\n".join(lines)

    @staticmethod
    def _knowledge_content(knowledge: Dict[str, Any]) -> str:
        if not knowledge:
            return ""
        if isinstance(knowledge.get("content"), list):
            return "\n".join(str(x) for x in knowledge["content"])
        if knowledge.get("content"):
            return str(knowledge["content"])
        return json.dumps(knowledge, ensure_ascii=False)

    @staticmethod
    def _prediction_text(item: Dict[str, Any]) -> str:
        for key in ["predict", "prediction", "output", "response", "generated_text", "text"]:
            if key in item:
                return item.get(key) or ""
        return ""

    @staticmethod
    def parse_conviction_output(text: str) -> Tuple[List[str], List[int]]:
        if not text:
            return [], []

        acc_match = re.search(r"罪名:\s*(\[.*?\])\s*;\s*法条:", text, re.S)
        law_match = re.search(r"法条:\s*(\[.*?\])", text, re.S)

        accusations = []
        laws = []
        if acc_match:
            try:
                parsed = ast.literal_eval(acc_match.group(1))
                if isinstance(parsed, list):
                    accusations = [str(x) for x in parsed]
            except (ValueError, SyntaxError):
                accusations = []

        if law_match:
            try:
                parsed = ast.literal_eval(law_match.group(1))
                if isinstance(parsed, list):
                    for item in parsed:
                        try:
                            laws.append(int(item))
                        except (TypeError, ValueError):
                            continue
            except (ValueError, SyntaxError):
                laws = []

        return accusations, laws

    @classmethod
    def load_conviction_predictions(
        cls,
        pred_path: str,
        sft_path: str
    ) -> Dict[Tuple[Any, str], Dict[str, Any]]:
        pred_data = cls.load_jsonl(pred_path) if pred_path.endswith(".jsonl") else cls.load_json(pred_path)
        sft_data = cls.load_json(sft_path)

        if len(pred_data) != len(sft_data):
            raise ValueError(
                f"Conviction prediction size {len(pred_data)} != SFT size {len(sft_data)}"
            )

        pred_map = {}
        for pred_item, sft_item in zip(pred_data, sft_data):
            case_id = sft_item.get("case_id")
            defendant = sft_item.get("defendant")
            if case_id is None or defendant is None:
                raise ValueError(
                    "Conviction SFT data must contain case_id and defendant. "
                    "Please regenerate it with Agent/ConvictionAgent.py."
                )
            accusations, laws = cls.parse_conviction_output(cls._prediction_text(pred_item))
            pred_map[(case_id, defendant)] = {
                "accusations": accusations,
                "laws": laws
            }
        return pred_map

    @staticmethod
    def load_json(file_path: str) -> List[Dict[str, Any]]:
        with open(file_path, "r", encoding="utf-8") as f:
            return json.load(f)

    @staticmethod
    def apply_conviction_predictions(
        dataset: List[Dict[str, Any]],
        pred_map: Dict[Tuple[Any, str], Dict[str, Any]]
    ) -> List[Dict[str, Any]]:
        updated_dataset = copy.deepcopy(dataset)
        for idx, case in enumerate(updated_dataset):
            case_id = case.get("id", idx)
            for defendant, info in case.get("criminals_info", {}).items():
                prediction = pred_map.get((case_id, defendant))
                if prediction is None:
                    info["accusations"] = []
                    info["laws"] = []
                    continue
                info["accusations"] = prediction.get("accusations", [])
                info["laws"] = prediction.get("laws", [])
        return updated_dataset

    def format_retrieved_knowledge(self, retrieval: Dict[str, Any]) -> Tuple[str, str, str]:
        general_law = retrieval.get("general_knowledge", {}).get("content", "")
        specific_parts = []

        for jurisdiction, charge_map in retrieval.get("knowledge", {}).items():
            for charge, knowledge in charge_map.items():
                content = self._knowledge_content(knowledge)
                if not content:
                    continue
                block = f"<{jurisdiction}-{charge}>\n{content}"
                specific_parts.append(block)

        if not general_law:
            general_law = (
                "量刑时应结合犯罪事实、性质、情节和社会危害程度，依次确定量刑起点、"
                "基准刑和宣告刑，并综合考虑从重、从轻、减轻等量刑情节。"
            )

        specific_law = "\n\n".join(specific_parts)
        charges = ",".join(retrieval.get("charges", {}).get("normalized", []))
        return general_law, specific_law, charges

    def build_input(self, case: Dict[str, Any], include_penalty: bool = True) -> str:
        retrieval = self.retriever.retrieve_legal_knowledge(case)
        general_law, specific_law, _ = self.format_retrieved_knowledge(retrieval)
        replacements = {
            "{demo}": self.PROMPT_GENERATE_GRAPH_DEMO,
            "{case_fact}": case.get("fact", ""),
            "{explanation}": case.get("interpretation", ""),
            "{sentence_result}": self.build_sentence_result(case, include_penalty=include_penalty),
            "{general_law}": general_law,
            "{specific_law}": specific_law
        }
        prompt = self.PROMPT_GENERATE_GRAPH
        for key, value in replacements.items():
            prompt = prompt.replace(key, value)
        return prompt

    def convert_prompt_data(
        self,
        dataset: List[Dict[str, Any]],
        include_penalty: bool = True
    ) -> List[Dict[str, Any]]:
        sft_data = []
        for case in tqdm(dataset, desc="Converting LegalKnow prompt data"):
            piece = {
                "instruction": self.SYSTEM_PROMPT,
                "input": self.build_input(case, include_penalty=include_penalty),
                "output": "",
                "history": [],
                "case_id": case.get("id")
            }
            sft_data.append(piece)
        return sft_data

    def generate_one(
        self,
        case: Dict[str, Any],
        model: str = "qwen-max",
        include_penalty: bool = True
    ) -> Dict[str, Any]:
        prompt = self.build_input(case, include_penalty=include_penalty)
        response, cost = ask_tyqw(prompt, model=model)
        return {
            "case_id": case.get("id"),
            "graph": response,
            "input_prompt": prompt,
            "cost": cost,
            "success": response is not None
        }

    def generate_graphs(
        self,
        dataset: List[Dict[str, Any]],
        output_path: str,
        model: str = "qwen-max",
        include_penalty: bool = True,
        max_workers: int = 8
    ):
        os.makedirs(os.path.dirname(output_path), exist_ok=True)
        with open(output_path, "w", encoding="utf-8"):
            pass
        results = []

        with ThreadPoolExecutor(max_workers=max_workers) as executor:
            futures = {
                executor.submit(self.generate_one, case, model, include_penalty): case
                for case in dataset
            }
            for future in tqdm(as_completed(futures), total=len(futures), desc="Generating LELG"):
                result = future.result()
                results.append(result)
                with open(output_path, "a", encoding="utf-8") as f:
                    f.write(json.dumps(result, ensure_ascii=False) + "\n")

        return results

    def prepare_llamafactory_prompt_data(
        self,
        train_path: str,
        valid_path: str,
        test_path: str,
        output_dir: str,
        dataset_info_path: str,
        valid_conviction_pred_path: Optional[str] = None,
        valid_conviction_sft_path: Optional[str] = None,
        test_conviction_pred_path: Optional[str] = None,
        test_conviction_sft_path: Optional[str] = None
    ):
        train_data = self.load_jsonl(train_path)
        valid_data = self.load_jsonl(valid_path)
        test_data = self.load_jsonl(test_path)

        if valid_conviction_pred_path:
            pred_map = self.load_conviction_predictions(
                valid_conviction_pred_path,
                valid_conviction_sft_path
            )
            valid_data = self.apply_conviction_predictions(valid_data, pred_map)

        if test_conviction_pred_path:
            pred_map = self.load_conviction_predictions(
                test_conviction_pred_path,
                test_conviction_sft_path
            )
            test_data = self.apply_conviction_predictions(test_data, pred_map)

        train_sft = self.convert_prompt_data(train_data, include_penalty=True)
        valid_sft = self.convert_prompt_data(valid_data, include_penalty=False)
        test_sft = self.convert_prompt_data(test_data, include_penalty=False)

        files = {
            "hrn_legalknow_train_prompt": "hrn_legalknow_train_prompt.json",
            "hrn_legalknow_valid_prompt": "hrn_legalknow_valid_prompt.json",
            "hrn_legalknow_test_prompt": "hrn_legalknow_test_prompt.json"
        }
        self.save_json(train_sft, os.path.join(output_dir, files["hrn_legalknow_train_prompt"]))
        self.save_json(valid_sft, os.path.join(output_dir, files["hrn_legalknow_valid_prompt"]))
        self.save_json(test_sft, os.path.join(output_dir, files["hrn_legalknow_test_prompt"]))

        for dataset_name, file_name in files.items():
            self.update_dataset_info(dataset_info_path, dataset_name, file_name)

        print("LegalKnow prompt data prepared successfully.")
        print(f"Train prompts: {len(train_sft)}")
        print(f"Valid prompts: {len(valid_sft)}")
        print(f"Test prompts: {len(test_sft)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare prompts or generate LELG with LegalKnow Assistant.")
    parser.add_argument("--mode", choices=["prepare", "generate"], default="prepare")
    parser.add_argument("--split", choices=["train", "valid", "test"], default="test")
    parser.add_argument("--model", default="qwen-max")
    parser.add_argument("--max-workers", type=int, default=8)
    parser.add_argument("--kb-path", default=None)
    parser.add_argument("--general-knowledge-path", default=None)
    parser.add_argument("--output-path", default=None)
    parser.add_argument("--conviction-pred-path", default=None)
    parser.add_argument("--conviction-sft-path", default=None)
    parser.add_argument("--valid-conviction-pred-path", default=None)
    parser.add_argument("--valid-conviction-sft-path", default=None)
    parser.add_argument("--test-conviction-pred-path", default=None)
    parser.add_argument("--test-conviction-sft-path", default=None)
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    paths = {
        "train": os.path.join(project_root, "dataset/HRN/data_train_v5.03.jsonl"),
        "valid": os.path.join(project_root, "dataset/HRN/data_valid_v5.03.jsonl"),
        "test": os.path.join(project_root, "dataset/HRN/data_test_v5.03.jsonl")
    }

    output_dir = os.path.join(project_root, "processed_data")
    dataset_info_path = os.path.join(output_dir, "dataset_info.json")

    agent = LegalKnowAssistant(
        kb_path=args.kb_path,
        general_knowledge_path=args.general_knowledge_path
    )

    if args.mode == "prepare":
        agent.prepare_llamafactory_prompt_data(
            train_path=paths["train"],
            valid_path=paths["valid"],
            test_path=paths["test"],
            output_dir=output_dir,
            dataset_info_path=dataset_info_path,
            valid_conviction_pred_path=args.valid_conviction_pred_path,
            valid_conviction_sft_path=args.valid_conviction_sft_path,
            test_conviction_pred_path=args.test_conviction_pred_path,
            test_conviction_sft_path=args.test_conviction_sft_path
        )
    else:
        dataset = agent.load_jsonl(paths[args.split])
        if args.conviction_pred_path:
            pred_map = agent.load_conviction_predictions(
                args.conviction_pred_path,
                args.conviction_sft_path
            )
            dataset = agent.apply_conviction_predictions(dataset, pred_map)
        output_path = args.output_path or os.path.join(
            project_root, "outputs", "LegalKnow_Assistant", f"{args.split}_lelg.jsonl"
        )
        agent.generate_graphs(
            dataset=dataset,
            output_path=output_path,
            model=args.model,
            include_penalty=args.split == "train",
            max_workers=args.max_workers
        )
