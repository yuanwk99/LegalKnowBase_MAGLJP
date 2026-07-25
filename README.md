# LegalKnowBase_MAGLJP

This repository contains the source code for **MAGLJP**, a multi-agent framework for multi-defendant legal judgment prediction. The framework follows the pipeline described in the paper:

1. **Conviction Agent** predicts the charges and applicable law articles for each defendant.
2. **LegalKnow Assistant** retrieves sentencing-related legal knowledge and generates the Legal Event Logic Graph (LELG).
3. **Sentencing Agent** predicts the term of penalty for each defendant based on the case fact, predicted charges/law articles, and the generated LELG.

The repository does **not** include LLaMA-Factory or datasets. Please install [LLaMA-Factory](https://github.com/hiyouga/LLaMA-Factory) separately and download the datasets from their official sources.



## Datasets

We use two multi-defendant legal judgment prediction datasets.

**MultiLJP.** MultiLJP (Multi-defendant Legal Judgment Prediction)~\citep{lyu2023hrn} is constructed from published legal documents in China Judgements Online. Professional annotators manually produce law articles, charges, terms of penalty, criminal relationships, and sentencing circumstances for each defendant in multi-defendant cases. MultiLJP contains 18,968 criminal cases with explicit annotations of inter-defendant relationships and charge allocations.

**CAIL2024-DRDZ.** CAIL2024-DRDZ is a high-quality dataset provided for the Chinese AI and Law Challenge 2024 (CAIL2024) DRDZ track, containing 15,000 cases specifically curated for multi-defendant reasoning~\citep{huang2024cmdl}.

The datasets are not redistributed in this repository. Please download them independently and prepare them in the JSONL format expected by the agents. For the default scripts, place the MultiLJP-style files under:

```bash
dataset/HRN/
├── data_train_v5.03.jsonl
├── data_valid_v5.03.jsonl
└── data_test_v5.03.jsonl
```

Each case should contain at least:

```json
{
  "id": 0,
  "fact": "...",
  "interpretation": "...",
  "defendants": ["[defendantA]", "[defendantB]"],
  "criminals_info": {
    "[defendantA]": {
      "accusations": ["..."],
      "laws": [264],
      "term": "..."
    }
  }
}
```

If `id` is missing, the code assigns the JSONL line number as the fallback case id.


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



