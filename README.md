# HybridMDS — Frozen Fusion for Vietnamese Multi-Document Summarization

Source code for **Frozen Fusion**, a hybrid (extractive + abstractive) summarization model for the Vietnamese Multi-Document Summarization task, evaluated on the [VLSP 2022 dataset](https://aihub.vn/competitions/341). The model also supports English multi-document summarization via the **MultiNews** dataset.

---

## Overview

Frozen Fusion is a two-stage summarization pipeline:

1. **Extractive stage** — A graph-based model scores every sentence in a multi-document cluster. The graph encodes four levels of granularity (sentence → section → document → cluster) using stepwise Graph Attention Networks (GAT). Six parallel sequence-modelling modules (BiLSTM, CNN, LSTM→CNN, CNN→LSTM, LSTM‖CNN, Self-Attention) fuse local and global context before a final ensemble selects the most salient sentences.
2. **Abstractive stage** — The extracted sentences are fed to a fine-tuned **BartPho-syllable-base** model (for Vietnamese) or any seq2seq model to produce a fluent, coherent summary.

Contrastive learning (InfoNCE) is used during extractive training to align the document-graph representation with the golden summary.

<img src="VLSP Dataset/overall_architecture.jpg" width="800px">

---

## Project Structure

```
HybridMDS/
│
├── VLSP Dataset/                        # Vietnamese dataset & artefacts
│   ├── LDA_models.pkl                   # Trained LDA section-division models (per category)
│   ├── public.csv                       
│   ├── test_segmentedSumm.pkl
│   ├── testing_data_new.jsonl
│   ├── train_data_new.jsonl
│   ├── train_segmentedSumm.txt
│   ├── validation_data_new.jsonl
│   ├── validation_segmentedSumm.txt
│   └── vietnamese-stopwords-dash.txt
│
├── MultiNews_Dataset/                   # English MultiNews dataset & artefacts
│   ├── {200,100,150}_{train,val,test}.json
│   ├── {200,100,150}_{train,val,test}_processed.json
│   ├── {200,100,150}_{train,val,test}_secnum.pkl
│   ├── LDA_models_4top.pkl
│   ├── unique_sents.json
│   ├── sent_embeddings.pkl              # ModernBERT sentence embeddings
│   ├── {train,val,test}.src.cleaned
│   ├── {train,val,test}.tgt
│   ├── load_data_en.py                      # English dataset loader
│   └── prepare_data.py                      # MultiNews preprocessing + LDA + sentence embeddings
│
├── LLM_prompting/                       # LLM inference & evaluation utilities
│   ├── LLM_as_judge.py
│   ├── LLM_as_judge_save_responses.py
│   └── LLM_inference.py
│
│   # Core model files
├── frozen_fusion_model.py               # Frozen Fusion model definition
├── graph_layers.py                      # MLP, GAT, StepWiseGraphConvLayer
├── graph_models.py                      # Contrast_Encoder, End2End_Encoder, InfoNCE
│
│   # Data loading
├── load_data.py                         # Vietnamese dataset loader (GraphDataset, SummDataset)
├── trainLDA.py                          # Train LDA models for VLSP section division
│
│   # Training & inference
├── extractive_trainer.py                # Extractive model training loop
├── BartPho_rewrite_trainer.py           # BartPho abstractive fine-tuning
├── Frozen_Fusion_infer.py               # End-to-end inference API
├── run_functions.py                     # train_e2e / val_e2e / get_summary helpers
│
│   # Utilities
├── utils.py                             # ROUGE helpers, seed_everything
├── requirements.txt
└── README.md
```

---

## Architecture

```
Input cluster (N sentences across D documents)
        │
        ▼
 [PhoBERT / ModernBERT]   ← frozen sentence encoder
        │  sentence embeddings
        ▼
 ┌─────────────────────────────────────┐
 │  StepWiseGraphConvLayer  ×2          │
 │  Contrast_Encoder  (InfoNCE loss)    │
 │  End2End_Encoder   (BCE loss)        │
 └─────────────────────────────────────┘
        │  enriched graph features
        ▼
 ┌───────────────────────────────────────────────┐
 │  Six parallel sequence modules                │
 │  BiLSTM │ CNN │ LSTM→CNN │ CNN→LSTM           │
 │  LSTM‖CNN │ Self-Attention                    │
 │  (each fuses BERT + graph features)           │
 └───────────────────────────────────────────────┘
        │  ensemble scores per sentence
        ▼
  Greedy extractive selection 
        │  extracted sentences
        ▼
  BartPho / seq2seq rewriter
        │
        ▼
    Final summary
```

---

## Data Preparation

### Vietnamese (VLSP Dataset)

1. Pre-compute PhoBERT word vectors for every sentence (produces `{mode}_vec/{mode}_{clusterid}.pkl`) and tokenize the corpus using the script provided separately (see the paper for details).
2. Train the LDA section-division models:

```bash
python trainLDA.py
# Outputs: VLSP Dataset/LDA_models_4top.pkl
```

### English (MultiNews Dataset)

1. The [MultiNews dataset](https://huggingface.co/datasets/alexfabbri/multi_news) as `.src.cleaned` / `.tgt` files under `MultiNews_Dataset/`.
2. Run the full preprocessing pipeline:

```bash
python prepare_data.py
```

This script will:
- Filter and clean raw clusters
- Split into train (200) / val (100) / test (150) subsets
- Train an LDA model for section division and save section assignments
- Encode all unique sentences with `ModernBERT-base` and save embeddings to `sent_embeddings.pkl`

---

## Training

### Step 1 — Train LDA Section Divider

*(Vietnamese only — already covered above)*

```bash
python trainLDA.py
```

### Step 2 — Train the Extractive Model

Edit hyperparameters at the top of `extractive_trainer.py` (epochs, learning rate, hidden size, etc.), then run:

```bash
python extractive_trainer.py
```

Key training details:
- **Loss**: `BCEWithLogitsLoss` for sentence scoring + InfoNCE contrastive loss
- **Optimiser**: AdamW with `ReduceLROnPlateau` scheduler
- **Graph encoder freezing**: The graph encoder is frozen after the contrastive loss stops improving (patience=2), then the task-specific modules continue training

Training progress is logged to [Weights & Biases](https://wandb.ai). Set your API key in `extractive_trainer.py`.

### Step 3 — Train the Abstractive Rewriter

```bash
python BartPho_rewrite_trainer.py
```

Key training details:
- **Base model**: `vinai/bartpho-syllable-base`
- **Framework**: HuggingFace `Seq2SeqTrainer`
- **Metric**: ROUGE-2-F1 (used for best model selection)

---

## Inference

Use `Frozen_Fusion_infer.py` for end-to-end inference on a list of raw document texts:

```python
from Frozen_Fusion_infer import MultiDocSummarizationAPI

docs = [
    "Nội dung tài liệu 1...",
    "Nội dung tài liệu 2...",
]
summary = MultiDocSummarizationAPI(docs, extractive_ratio=5)
print(summary)
```

**Arguments:**

| Parameter | Description                                                | Default |
|---|------------------------------------------------------------|--------|
| `docs` | List of raw document strings                               | —      |
| `extractive_ratio` | Fraction or number of sentences to extract (by word count) |   —     |
| `golden` | Optional reference summary for ROUGE evaluation            | `None` |

The function automatically handles section division (LDA), graph construction, extractive scoring, and abstractive rewriting.

To generate summaries for the full VLSP test set and write `results.txt`:

```bash
python Frozen_Fusion_infer.py
# Output: output_summs/results.txt
```

Upload `results.txt` to [AI Hub](https://aihub.vn/competitions/341) for the official leaderboard evaluation.

---

## Evaluation

### Automatic Metrics (ROUGE)

ROUGE-1/2/L scores are computed automatically during training via `utils.cal_rouge`. For manual evaluation:

```python
from utils import cal_rouge
scores = cal_rouge(references, predictions)
# Returns (R1-P, R1-R, R1-F, R2-P, R2-R, R2-F, RL-P, RL-R, RL-F)
```

### LLM-as-Judge

The `LLM_prompting/` folder contains utilities to evaluate generated summaries along four dimensions — **Coherence**, **Consistency**, **Fluency**, **Relevance** — using an LLM judge (default: `Qwen/Qwen3-8B` via HuggingFace Inference API).

**Workflow:**

1. Generate summaries with `Frozen_Fusion_infer.py`.
2. Build evaluation prompts and optionally call the LLM automatically:

```bash
# From the project root
python LLM_prompting/LLM_as_judge.py
```

3. Parse and save scores:

```bash
python LLM_prompting/LLM_as_judge_save_responses.py
```

Scores are appended to an Excel file under `VLSP Dataset/LLM_summaries/`.

---

