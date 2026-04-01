# Agent-Event-Coder (AEC)

Partial code for the AAAI 2026 (Oral) paper:

> **Extracting Events Like Code: A Multi-Agent Programming Framework for Zero-Shot Event Extraction**  
> Quanjiang Guo, Sijie Wang, Jinchuan Zhang, Ben Zhang, Zhao Kang, Ling Tian, Ke Yan  
> UESTC · AAAI 2026 · [arXiv 2511.13118](https://arxiv.org/abs/2511.13118)

AEC treats zero-shot event extraction as a **code generation** problem. Four specialized LLM agents (Retrieval, Planning, Coding, Verification) collaborate in a dual-loop refinement algorithm to produce Python class instantiations as structured event outputs.

---

## Table of Contents

1. [Environment Setup](#1-environment-setup)
2. [Dataset Acquisition](#2-dataset-acquisition)
3. [TextEE Preprocessing](#3-textee-preprocessing)
4. [LLM Setup](#4-llm-setup)
5. [Running Inference](#5-running-inference)
6. [Evaluation](#6-evaluation)
7. [Repository Structure](#7-repository-structure)
8. [Citation](#8-citation)

---

## 1. Environment Setup

### 1.1 Create conda environment

```bash
conda create -n AEC python=3.10 -y
conda activate AEC
pip install -r requirements.txt
```

Contents of `requirements.txt`:
```
pydantic>=2.0.0
openai>=1.0.0
prettytable>=3.0.0
tqdm>=4.0.0
numpy>=1.20.0
datasets>=2.0.0
pandas>=1.3.0
scikit-learn>=1.0.0
```

---

## 2. Dataset Acquisition

The paper evaluates on **five datasets** from the TextEE benchmark.  
Raw data must be obtained independently; preprocessing is done via the [TextEE](https://github.com/ej0cl6/TextEE) framework.

| Dataset | Domain | Task | # Types | Freely Available? |
|---|---|---|---|---|
| `ace05-en` | News | E2E | 33 | ❌ Requires LDC license |
| `fewevent` | General | ED | 100 | ✅ Public |
| `genia2011` | Biomedical | E2E | 9 | ✅ Public |
| `speed` | Epidemiology | ED | 7 | ✅ Public |
| `casie` | Cybersecurity | E2E | 5 | ✅ Public |

---

### Dataset 1 — ACE05-EN (`ace05-en`) ⚠️ License Required

**Source**: LDC (Linguistic Data Consortium)  
**Catalog number**: [LDC2006T06](https://catalog.ldc.upenn.edu/LDC2006T06)  
**License**: LDC User Agreement — requires institutional or individual membership.

**Access steps**:
1. Go to https://catalog.ldc.upenn.edu/LDC2006T06
2. Register / log in to LDC (free academic registration available)
3. Purchase or request access (cost varies: ~$0 for LDC members, ~$50–150 for non-members)
4. Download the corpus archive; extract to a local directory

**What you get**: `~/ldc/ace05/` containing `.sgm` (source text) and `.apf.xml` (annotation) files organized by genre (`bc/`, `bn/`, `nw/`, `un/`, `wl/`, `cts/`).

> **Note**: If you are at a university with an LDC membership, you may access all LDC corpora for free through your institution. Check with your library or IT department.

---

### Dataset 2 — FewEvent (`fewevent`) ✅

**Source**: "Meta-Learning with Dynamic-Memory-Based Prototypical Network for Few-Shot Event Detection" (WSDM 2020)  
**Original repo**: https://github.com/thunlp/FewEvent

```bash
# Clone the FewEvent repository
git clone https://github.com/thunlp/FewEvent.git
# Raw data files are in FewEvent/data/
ls FewEvent/data/
# Expected: event_dict_data_dir/, train.json, dev.json, test.json
```

If the above URL is unavailable, the dataset is also mirrored on Hugging Face:
```python
from datasets import load_dataset
ds = load_dataset("willcb/few-event")
```

**Format**: Each record has `{"tokens": [...], "event_mentions": [{"event_type": "...", "trigger": {...}}]}`

---

### Dataset 3 — GENIA 2011 (`genia2011`) ✅

**Source**: BioNLP Shared Task 2011 — Genia Event (GE) task  
**Download page**: http://2011.bionlp-st.dbcls.jp/downloads  
**License**: Freely available for research use

```bash
# Create raw data directory
mkdir -p data/raw/genia2011

# Download training, development, and test sets
wget -P data/raw/genia2011 \
    http://2011.bionlp-st.dbcls.jp/GE11/downloads/BioNLP-ST_2011_genia_train_data_rev1.tar.gz \
    http://2011.bionlp-st.dbcls.jp/GE11/downloads/BioNLP-ST_2011_genia_devel_data_rev1.tar.gz \
    http://2011.bionlp-st.dbcls.jp/GE11/downloads/BioNLP-ST_2011_genia_test_data.tar.gz

# Extract
cd data/raw/genia2011
tar -xzf BioNLP-ST_2011_genia_train_data_rev1.tar.gz
tar -xzf BioNLP-ST_2011_genia_devel_data_rev1.tar.gz
tar -xzf BioNLP-ST_2011_genia_test_data.tar.gz
```

**Format**: BioNLP standoff format (`.a1` entity annotations, `.a2` event annotations, `.txt` source text).

---

### Dataset 4 — SPEED (`speed`) ✅

**Source**: "Event Detection from Social Media for Epidemic Prediction" (NAACL 2024)  
**Paper**: https://aclanthology.org/2024.naacl-long.438/

```bash
# The SPEED dataset can be obtained from the paper authors.
# Check the paper page or the supplementary materials for a data link.
# Alternatively, search the ACL Anthology supplementary material:
# https://aclanthology.org/2024.naacl-long.438.zip
```

If a GitHub repo is linked in the paper, clone it:
```bash
git clone https://github.com/<SPEED-authors>/SPEED.git   # check paper for exact URL
```

**Format**: JSON lines with `{"text": "...", "event_mentions": [...]}`  
**Event types**: 7 epidemic-related types (Spread, Prevention, Symptoms, Treatment, etc.)

---

### Dataset 5 — CASIE (`casie`) ✅

**Source**: "CASIE: Extracting Cybersecurity Event Information from Text" (AAAI 2020)  
**GitHub**: https://github.com/Ebiquity/CASIE  
**License**: Freely available

```bash
git clone https://github.com/Ebiquity/CASIE.git
ls CASIE/data/
# annotation/  source/
```

**Format**: JSON annotation files + plain text source files.  
**Event types**: 5 cybersecurity types (Databreach, Ransom, Phishing, DiscoverVulnerability, PatchVulnerability).

---

## 3. TextEE Preprocessing

All five datasets must be converted to TextEE's standardized JSON-lines format before inference. TextEE provides preprocessors for each dataset.

### 3.1 Install TextEE

```bash
git clone https://github.com/ej0cl6/TextEE.git
cd TextEE
pip install -r requirements.txt   # or: conda env create -f env.yml
python -m spacy download en_core_web_lg
```

### 3.2 Expected output structure

After preprocessing, TextEE produces split files at:
```
TextEE/data/
└── {dataset}/
    └── split1/
        ├── train.json
        ├── dev.json
        └── test.json
```

Each line in these files is a JSON object:
```json
{
  "wnd_id": "doc_id-sentence_id",
  "text": "The soldiers attacked the village.",
  "event_mentions": [
    {
      "event_type": "Conflict:Attack",
      "trigger": {"text": "attacked", "start": 13, "end": 21},
      "arguments": [
        {"role": "Attacker", "text": "soldiers", "start": 4, "end": 12}
      ]
    }
  ]
}
```

### 3.3 Run preprocessing

```bash
cd TextEE

# ACE05-EN (replace path with your LDC data location)
python TextEE/preprocess.py \
    --dataset ACE05 \
    --input_dir /path/to/ldc/ace05 \
    --output_dir data/ace05-en

# FewEvent
python TextEE/preprocess.py \
    --dataset FewEvent \
    --input_dir /path/to/FewEvent/data \
    --output_dir data/fewevent

# GENIA2011
python TextEE/preprocess.py \
    --dataset Genia2011 \
    --input_dir /path/to/genia2011/raw \
    --output_dir data/genia2011

# SPEED
python TextEE/preprocess.py \
    --dataset SPEED \
    --input_dir /path/to/SPEED/data \
    --output_dir data/speed

# CASIE
python TextEE/preprocess.py \
    --dataset CASIE \
    --input_dir /path/to/CASIE/data \
    --output_dir data/casie
```

> **Note**: The exact `--dataset` flag names and argument structure may vary. Check `TextEE/preprocess.py --help` for the current interface. If the script is not at that path, look for it under `TextEE/TextEE/preprocess.py`.

### 3.4 Copy preprocessed data to AEC

```bash
# From the TextEE directory:
mkdir -p /home/users/yy/code/AEC/data/raw/TextEE

cp -r data/ace05-en  /home/users/yy/code/AEC/data/raw/TextEE/
cp -r data/fewevent  /home/users/yy/code/AEC/data/raw/TextEE/
cp -r data/genia2011 /home/users/yy/code/AEC/data/raw/TextEE/
cp -r data/speed     /home/users/yy/code/AEC/data/raw/TextEE/
cp -r data/casie     /home/users/yy/code/AEC/data/raw/TextEE/
```

Final structure expected by `run_inference.py`:
```
AEC/data/raw/TextEE/
├── ace05-en/split1/{train,dev,test}.json
├── fewevent/split1/{train,dev,test}.json
├── genia2011/split1/{train,dev,test}.json
├── speed/split1/{train,dev,test}.json
└── casie/split1/{train,dev,test}.json
```

---

## 4. LLM Setup

### Option A — OpenAI API (GPT-4o / GPT-3.5-turbo)

Best results. Requires an API key.

```bash
export OPENAI_API_KEY=sk-...
```

### Option B — Local via vLLM (Recommended for open-source models)

Serves any HuggingFace model behind an OpenAI-compatible HTTP endpoint.  
Matches the paper's setup (Llama3-8B-Instruct, Llama3-70B-Instruct).

**Requirements**: GPU with ≥16 GB VRAM for the 8B model (≥80 GB total for 70B across 4 GPUs).

```bash
pip install vllm

# Get a HuggingFace token for gated models (free):
# https://huggingface.co/settings/tokens
# Then accept the Llama 3 license at:
# https://huggingface.co/meta-llama/Meta-Llama-3-8B-Instruct

# Start vLLM server (run in a separate terminal / tmux)
HF_TOKEN=hf_... \
vllm serve meta-llama/Meta-Llama-3-8B-Instruct \
    --port 8000 \
    --tensor-parallel-size 1      # use 4 to match the paper (70B)

# Verify the server is up
curl http://localhost:8000/v1/models
```

### Option C — Local via Ollama (Easiest setup)

Uses 4-bit quantized models. Slightly lower quality than full precision but very easy to run, even on CPU.

```bash
# Install Ollama: https://ollama.com
curl -fsSL https://ollama.com/install.sh | sh

# Pull Llama 3.1 8B (quantized, ~5 GB)
ollama pull llama3.1:8b

# Ollama serves at http://localhost:11434 by default
# (auto-starts when you run a model)
```

### Comparison

| | OpenAI API | vLLM | Ollama |
|---|---|---|---|
| Model | GPT-4o / GPT-3.5 | Llama-3-8B (exact paper model) | llama3.1:8b (4-bit) |
| Matches paper | Best scores | ✅ Exact match | Close |
| GPU needed | None | ≥16 GB VRAM | Optional |
| HF token needed | No | Yes (gated model) | No |
| Setup effort | Low | Medium | Very low |

---

## 5. Running Inference

All commands below are run from the **AEC directory** (`/home/users/yy/code/AEC`).

### 5.1 Full pipeline on a dataset

```bash
conda activate AEC
cd /home/users/yy/code/AEC

# --- GPT-4o ---
OPENAI_API_KEY=sk-... \
python run_inference.py \
    --dataset ace05-en \
    --model gpt-4o \
    --k 3 --t 3

# --- Llama-3-8B via vLLM (start server first, see §4B) ---
OPENAI_API_KEY=EMPTY \
python run_inference.py \
    --dataset ace05-en \
    --model meta-llama/Meta-Llama-3-8B-Instruct \
    --base_url http://localhost:8000/v1 \
    --k 3 --t 3

# --- Llama-3.1-8B via Ollama ---
OPENAI_API_KEY=ollama \
python run_inference.py \
    --dataset casie \
    --model llama3.1:8b \
    --base_url http://localhost:11434/v1 \
    --k 3 --t 3

# --- GPT-3.5-turbo (cheaper, lower scores) ---
OPENAI_API_KEY=sk-... \
python run_inference.py \
    --dataset fewevent \
    --model gpt-3.5-turbo \
    --k 3 --t 3
```

### 5.2 All five datasets (paper reproduction)

```bash
#!/bin/bash
# Run from: /home/users/yy/code/AEC

DATASETS=("ace05-en" "fewevent" "genia2011" "speed" "casie")
MODEL="meta-llama/Meta-Llama-3-8B-Instruct"
BASE_URL="http://localhost:8000/v1"

for DATASET in "${DATASETS[@]}"; do
    echo "=== Running $DATASET ==="
    OPENAI_API_KEY=EMPTY python run_inference.py \
        --dataset "$DATASET" \
        --model "$MODEL" \
        --base_url "$BASE_URL" \
        --k 3 --t 3 \
        --resume    # safe to re-run; skips already-done samples
done
```

### 5.3 CLI flags reference

| Flag | Default | Description |
|---|---|---|
| `--dataset` | required | One of: `ace05-en`, `fewevent`, `genia2011`, `speed`, `casie` |
| `--model` | `gpt-4o` | LLM model ID |
| `--base_url` | OpenAI | Point to vLLM/Ollama server |
| `--k` | `3` | Max planning hypotheses (paper: 3) |
| `--t` | `3` | Max patch attempts per hypothesis (paper: 3) |
| `--split` | `test` | Dataset split: `train`, `dev`, or `test` |
| `--max_samples` | all | Limit to first N samples (for quick testing) |
| `--output` | auto | Output JSON path |
| `--delay` | `1.0` | Sleep seconds between API calls (rate limit) |
| `--resume` | off | Skip already-processed samples |
| `--no_eval` | off | Skip auto-evaluation after inference |
| `--eval_only FILE` | — | Only evaluate an existing predictions file |

---

## 6. Evaluation

Evaluation runs automatically after inference. Predictions are saved to:
```
AEC/outputs/{dataset}_{model}_predictions.json
AEC/outputs/{dataset}_{model}_predictions_scores.json   ← F1 scores (×100)
```

### 6.1 Manually evaluate an existing predictions file

```bash
cd /home/users/yy/code/AEC
python run_inference.py \
    --eval_only outputs/ace05-en_gpt4o_predictions.json
```

Or run the scorer directly:
```bash
cd /home/users/yy/code/AEC/utils/code_evaluation
python events_scorer.py \
    --input_file ../../outputs/ace05-en_gpt4o_predictions.json
```

### 6.2 Metrics

| Metric | Description |
|---|---|
| **TI** — Trigger Identification | F1 for correct trigger span (any event type) |
| **TC** — Trigger Classification | F1 for correct trigger span + correct event type |
| **AI** — Argument Identification | F1 for correct argument span (linked to correct trigger) |
| **AC** — Argument Classification | F1 for correct argument span + correct role |

All metrics are **micro-averaged** over three independent runs in the paper.

### 6.3 Reference results (from paper Table 1, Llama3-8B-Instruct)

| Dataset | Task | TI | TC | AI | AC |
|---|---|---|---|---|---|
| `ace05-en` | E2E | 56.6 | 48.8 | 37.6 | 34.2 |
| `fewevent` | ED | 38.4 | 36.5 | — | — |
| `genia2011` | E2E | 47.2 | 40.1 | 33.7 | 28.9 |
| `speed` | ED | 61.8 | 58.3 | — | — |
| `casie` | E2E | 66.3 | 63.1 | 34.8 | 31.2 |

---

## 7. Repository Structure

```
AEC/
├── __init__.py                  # Package init; exports main classes
├── run_inference.py             # ★ Full paper reproduction script
├── planning_agent.py            # Agent 2: trigger hypothesis generation (LLM)
├── coding_agent.py              # Agent 3: Python code generation (LLM)
├── retrieval_agent.py           # Agent 1: exemplar sentence generation (LLM)
├── verification_agent.py        # Agent 4: T1/T2/T3 verification checks
├── ontology.py                  # Event schema manager
├── event_schema.py              # EventSchema / EventObject data classes
├── llm_utils.py                 # OpenAI / vLLM API wrapper
├── requirements.txt
│
├── data/
│   └── raw/TextEE/              # ← place preprocessed TextEE data here
│       ├── ace05-en/split1/
│       ├── fewevent/split1/
│       ├── genia2011/split1/
│       ├── speed/split1/
│       └── casie/split1/
│
├── outputs/                     # Inference predictions + scores saved here
│
└── utils/
    ├── code_evaluation/
    │   ├── events_scorer.py     # Precision / Recall / F1 scorer
    │   ├── all_ee_definitions.py
    │   └── utils_typing.py
    ├── code_prompts/
    │   ├── prepare_dataset.py   # Convert raw TextEE data to code prompts
    │   └── utils.py
    └── code_schema_generation/
        ├── python_event_defs/   # Python dataclass definitions per dataset
        │   ├── ace05-en_definitions_new.py
        │   ├── casie_definitions_new.py
        │   ├── fewevent_definitions_new.py
        │   ├── genia2011_definitions_new.py
        │   └── speed_definitions_new.py
        └── init_prompts/        # Schema strings used as LLM context
            ├── ace05-en.txt
            ├── casie.txt
            ├── fewevent.txt
            ├── genia2011.txt
            └── speed.txt
```

---

## 8. Citation

If you use this code or the AEC framework, please cite:

```bibtex
@inproceedings{guo2026aec,
  title     = {Extracting Events Like Code: A Multi-Agent Programming Framework
               for Zero-Shot Event Extraction},
  author    = {Quanjiang Guo and Sijie Wang and Jinchuan Zhang and Ben Zhang and
               Zhao Kang and Ling Tian and Ke Yan},
  booktitle = {Proceedings of the 40th AAAI Conference on Artificial Intelligence},
  year      = {2026},
}
```

Also cite TextEE if you use their preprocessing:

```bibtex
@article{huang2023textee,
  title   = {TextEE: Benchmark, Reevaluation, Reflections, and Future Challenges
             in Event Extraction},
  author  = {Kuan-Hao Huang and I-Hung Hsu and Tanmay Parekh and Zhiyu Xie and
             Zixuan Zhang and Premkumar Natarajan and Kai-Wei Chang and
             Nanyun Peng and Heng Ji},
  journal = {arXiv preprint arXiv:2311.09562},
  year    = {2023},
}
```
