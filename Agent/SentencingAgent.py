import argparse
import ast
import copy
import json
import os
import re
import sys
from typing import Any, Dict, List, Optional

from tqdm import tqdm

PROJECT_ROOT = os.path.abspath(os.path.join(os.path.dirname(__file__), ".."))
if PROJECT_ROOT not in sys.path:
    sys.path.insert(0, PROJECT_ROOT)

from Agent.LegalKnowAssistant import LegalKnowAssistant


class SentencingAgent:
    SYSTEM_PROMPT = "你是一个拥有丰富法律知识并且精通罪名预测的专家。"

    PROMPT_SENTENCE = """你需要根据<犯罪事实>和刑期知识形成的刑期因果图谱，给出{name}在此案中涉及的刑期。
已知{name}触犯刑法第{laws}条,构成{accusations}。
[要求]仅输出刑期，格式例如：'有期徒刑X年'。不要生成任何解释！

<犯罪事实>
{case_fact}
</犯罪事实>

<刑期因果图谱>
{lelg}
</刑期因果图谱>"""

    def __init__(self):
        pass

    @staticmethod
    def load_jsonl(file_path: str) -> List[Dict[str, Any]]:
        data = []
        if not file_path or not os.path.exists(file_path):
            return data
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
    def _extract_python_block(text: str) -> str:
        if not text:
            return ""
        match = re.search(r"```(?:python)?\s*(.*?)```", text, re.S)
        return match.group(1).strip() if match else text.strip()

    @staticmethod
    def strip_penalty_nodes(lelg: str) -> str:
        """
        Remove p-type penalty result nodes from nodes to avoid leaking gold terms.
        Edges are intentionally kept unchanged, following the project setting.
        """
        code = SentencingAgent._extract_python_block(lelg)
        if not code:
            return ""

        try:
            module = ast.parse(code)
        except SyntaxError:
            return lelg

        nodes = None
        edges = None
        for stmt in module.body:
            if not isinstance(stmt, ast.Assign):
                continue
            for target in stmt.targets:
                if isinstance(target, ast.Name) and target.id == "nodes":
                    nodes = ast.literal_eval(stmt.value)
                elif isinstance(target, ast.Name) and target.id == "edges":
                    edges = ast.literal_eval(stmt.value)

        if nodes is None or edges is None:
            return lelg

        if not isinstance(nodes, dict) or not isinstance(edges, list):
            return lelg

        filtered_nodes = {
            node_id: value
            for node_id, value in nodes.items()
            if not (isinstance(value, dict) and value.get("type") == "p")
        }

        return (
            "nodes = "
            + json.dumps(filtered_nodes, ensure_ascii=False, indent=4)
            + "\n\nedges = "
            + json.dumps(edges, ensure_ascii=False, indent=4)
        )

    @staticmethod
    def load_graph_map(graph_path: Optional[str], strip_penalty_nodes: bool = True) -> Dict[Any, str]:
        if not graph_path or not os.path.exists(graph_path):
            return {}

        graph_map = {}
        with open(graph_path, "r", encoding="utf-8") as f:
            for idx, line in enumerate(tqdm(f.readlines(), desc=f"Loading {os.path.basename(graph_path)}")):
                line = line.strip()
                if not line:
                    continue
                item = json.loads(line)

                if isinstance(item, dict):
                    case_id = item.get("case_id")
                    if case_id is None:
                        case_id = item.get("id", idx)
                    graph = item.get("graph", item.get("lelg", item.get("output", "")))
                    if graph:
                        graph_map[case_id] = (
                            SentencingAgent.strip_penalty_nodes(graph)
                            if strip_penalty_nodes else graph
                        )
                elif isinstance(item, list) and len(item) >= 2:
                    case = item[0][0] if isinstance(item[0], list) and item[0] else item[0]
                    case_id = case.get("id", idx) if isinstance(case, dict) else idx
                    graph_map[case_id] = (
                        SentencingAgent.strip_penalty_nodes(item[1])
                        if strip_penalty_nodes else item[1]
                    )

        return graph_map

    @staticmethod
    def apply_conviction_predictions(
        dataset: List[Dict[str, Any]],
        pred_path: Optional[str],
        sft_path: Optional[str]
    ) -> List[Dict[str, Any]]:
        if not pred_path:
            return dataset
        pred_map = LegalKnowAssistant.load_conviction_predictions(pred_path, sft_path)
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

    def build_input(self, case: Dict[str, Any], name: str, lelg: str) -> str:
        info = case.get("criminals_info", {}).get(name, {})
        laws = "，".join(str(x) for x in info.get("laws", []))
        accusations = "，".join(str(x) for x in info.get("accusations", []))
        return self.PROMPT_SENTENCE.format(
            name=name,
            laws=laws,
            accusations=accusations,
            case_fact=case.get("fact", ""),
            lelg=lelg or ""
        )

    @staticmethod
    def build_output(case: Dict[str, Any], name: str) -> str:
        return str(case.get("criminals_info", {}).get(name, {}).get("term", ""))

    def convert_sft_data(
        self,
        dataset: List[Dict[str, Any]],
        graph_map: Optional[Dict[Any, str]] = None,
        skip_without_graph: bool = False
    ) -> List[Dict[str, Any]]:
        graph_map = graph_map or {}
        sft_data = []

        for idx, case in enumerate(tqdm(dataset, desc="Converting Sentencing SFT data")):
            case_id = case.get("id", idx)
            lelg = graph_map.get(case_id, case.get("lelg", case.get("graph", "")))
            lelg = self.strip_penalty_nodes(lelg)
            if skip_without_graph and not lelg:
                continue

            defendants = case.get("defendants") or list(case.get("criminals_info", {}).keys())
            for name in defendants:
                piece = {
                    "instruction": self.SYSTEM_PROMPT,
                    "input": self.build_input(case, name, lelg),
                    "output": self.build_output(case, name),
                    "history": [],
                    "case_id": case_id,
                    "defendant": name
                }
                sft_data.append(piece)

        return sft_data

    def prepare_llamafactory_data(
        self,
        train_path: str,
        valid_path: str,
        test_path: str,
        output_dir: str,
        dataset_info_path: str,
        train_graph_path: Optional[str] = None,
        valid_graph_path: Optional[str] = None,
        test_graph_path: Optional[str] = None,
        valid_conviction_pred_path: Optional[str] = None,
        valid_conviction_sft_path: Optional[str] = None,
        test_conviction_pred_path: Optional[str] = None,
        test_conviction_sft_path: Optional[str] = None,
        skip_without_graph: bool = False
    ):
        train_data = self.load_jsonl(train_path)
        valid_data = self.load_jsonl(valid_path)
        test_data = self.load_jsonl(test_path)

        valid_data = self.apply_conviction_predictions(
            valid_data,
            valid_conviction_pred_path,
            valid_conviction_sft_path
        )
        test_data = self.apply_conviction_predictions(
            test_data,
            test_conviction_pred_path,
            test_conviction_sft_path
        )

        train_graph = self.load_graph_map(train_graph_path)
        valid_graph = self.load_graph_map(valid_graph_path)
        test_graph = self.load_graph_map(test_graph_path)

        train_sft = self.convert_sft_data(train_data, train_graph, skip_without_graph)
        valid_sft = self.convert_sft_data(valid_data, valid_graph, skip_without_graph)
        test_sft = self.convert_sft_data(test_data, test_graph, skip_without_graph)

        files = {
            "hrn_sentencing_train_sft": "hrn_sentencing_train_sft.json",
            "hrn_sentencing_valid_sft": "hrn_sentencing_valid_sft.json",
            "hrn_sentencing_test_sft": "hrn_sentencing_test_sft.json"
        }

        self.save_json(train_sft, os.path.join(output_dir, files["hrn_sentencing_train_sft"]))
        self.save_json(valid_sft, os.path.join(output_dir, files["hrn_sentencing_valid_sft"]))
        self.save_json(test_sft, os.path.join(output_dir, files["hrn_sentencing_test_sft"]))

        for dataset_name, file_name in files.items():
            self.update_dataset_info(dataset_info_path, dataset_name, file_name)

        print("Sentencing Agent data prepared successfully.")
        print(f"Train samples: {len(train_sft)}")
        print(f"Valid samples: {len(valid_sft)}")
        print(f"Test samples: {len(test_sft)}")


def parse_args():
    parser = argparse.ArgumentParser(description="Prepare LLaMA-Factory SFT data for Sentencing Agent.")
    parser.add_argument("--train-graph", default=None)
    parser.add_argument("--valid-graph", default=None)
    parser.add_argument("--test-graph", default=None)
    parser.add_argument("--valid-conviction-pred-path", default=None)
    parser.add_argument("--valid-conviction-sft-path", default=None)
    parser.add_argument("--test-conviction-pred-path", default=None)
    parser.add_argument("--test-conviction-sft-path", default=None)
    parser.add_argument("--skip-without-graph", action="store_true")
    return parser.parse_args()


if __name__ == "__main__":
    args = parse_args()
    current_dir = os.path.dirname(os.path.abspath(__file__))
    project_root = os.path.abspath(os.path.join(current_dir, ".."))

    train_path = os.path.join(project_root, "dataset/HRN/data_train_v5.03.jsonl")
    valid_path = os.path.join(project_root, "dataset/HRN/data_valid_v5.03.jsonl")
    test_path = os.path.join(project_root, "dataset/HRN/data_test_v5.03.jsonl")

    output_dir = os.path.join(project_root, "processed_data")
    dataset_info_path = os.path.join(output_dir, "dataset_info.json")

    agent = SentencingAgent()
    agent.prepare_llamafactory_data(
        train_path=train_path,
        valid_path=valid_path,
        test_path=test_path,
        output_dir=output_dir,
        dataset_info_path=dataset_info_path,
        train_graph_path=args.train_graph,
        valid_graph_path=args.valid_graph,
        test_graph_path=args.test_graph,
        valid_conviction_pred_path=args.valid_conviction_pred_path,
        valid_conviction_sft_path=args.valid_conviction_sft_path,
        test_conviction_pred_path=args.test_conviction_pred_path,
        test_conviction_sft_path=args.test_conviction_sft_path,
        skip_without_graph=args.skip_without_graph
    )
