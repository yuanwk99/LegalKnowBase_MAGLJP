# LegalKnowBase_MAGLJP

This repository contains the source code for **MAGLJP**, a multi-agent framework for multi-defendant legal judgment prediction. The framework follows the pipeline described in the paper:

1. **Conviction Agent** predicts the charges and applicable law articles for each defendant.
2. **LegalKnow Assistant** retrieves sentencing-related legal knowledge and generates the Legal Event Logic Graph (LELG).
3. **Sentencing Agent** predicts the term of penalty for each defendant based on the case fact, predicted charges/law articles, and the generated LELG.

The repository does **not** include LLaMA-Factory or datasets. Please install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) separately and download the datasets from their official sources.



## Datasets

We use two multi-defendant legal judgment prediction datasets.

**MultiLJP.** 

**CAIL2024-DRDZ.** 

The datasets are not redistributed in this repository. Please download them independently and prepare them in the JSONL format expected by the agents. For the default scripts, place the MultiLJP-style files under:

```bash
dataset/HRN/
├── data_train_v5.03.jsonl
├── data_valid_v5.03.jsonl
└── data_test_v5.03.jsonl
```


## Environment

Install the lightweight dependencies used by this repository:

```bash
pip install tqdm requests
```

Install DashScope only if you need to call Qwen-max to generate LELG:

```bash
pip install dashscope
export DASHSCOPE_API_KEY=your_key
```

Install LLaMA-Factory separately and make sure `llamafactory-cli` is available.

In the commands below, `/path/to/IPM-src` denotes the absolute path of this repository and `/path/to/Qwen2.5-7B-Instruct` denotes the base model path.

