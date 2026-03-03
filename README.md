# Bloom AI

**Official project title (Form 1):**  
*BloomAI: An LLM-Powered System for Cognitively Aligned Assessment Design Using Bloom’s Taxonomy for Question Classification and Generation*


- **Academic Advisor:** Dr. Omar Alomeir  
- **Contributors:** Saleh Albalawi  

Bloom AI is an LLM-powered assessment design system that generates and evaluates questions aligned with Bloom’s Taxonomy, with explainable cognitive-level classification. This repository implements the system described in the course Form 1 project sign-up and associated documentation.

---

## How This Was Created (CS499)

This section documents how the binary question-quality classifier was added, how the pipeline was orchestrated, and the results.

### Requirements (Doctor / CS499)

- The system must **learn from the dataset**: a supervised binary classifier is trained on labeled data.
- The dataset has schema: `question`, `label`, `is_labeled`, `source`, `dataset_type`. **Label** is binary: `1` = valid/content-related, `0` = invalid/meta/vague. These labels are **not** Bloom-level labels and must not be treated as such.
- The trained component must be **integrated into the generation pipeline**: the classifier runs **after** LLM generation and filters out questions predicted as invalid.
- No Bloom-label training, no LLM fine-tuning, no removal of dataset columns. Focus: binary classification + pipeline integration.

### Changes We Made

1. **Config** (`app/config.py`)  
   - Added `DATASET_PATH` (CSV path, overridable via `BLOOM_DATASET_PATH`) and `QUALITY_MODEL_PATH` (saved classifier).

2. **Training script** (`scripts/train_quality_classifier.py`)  
   - Loads the CSV (columns: `question`, `label`, `is_labeled`, `source`, `dataset_type`).  
   - Uses only rows with binary `label` 0 or 1.  
   - Splits data: **70% train**, **15% validation**, **15% test** (stratified).  
   - Trains a pipeline: **TF-IDF** (max 5000 features, unigrams + bigrams) + **LogisticRegression**.  
   - Reports **accuracy**, **precision**, **recall**, **F1** on validation and test.  
   - Saves the pipeline as `models/quality_classifier.joblib` and metrics as `models/quality_classifier_metrics.json`.

3. **Inference module** (`app/quality/classifier.py`)  
   - Loads the saved model (lazy).  
   - `predict(question)` → 0 or 1.  
   - `filter_valid_items(items, question_key="question")` → keeps only items whose question is predicted as 1 (valid).

4. **LLM generator** (`app/llm/generator.py`)  
   - `generate_questions(topic, bloom, qtype, n)` calls Azure OpenAI and returns a list of item dicts (question, choices, etc.). Bloom level is **prompt-only**; no Bloom classification is trained.

5. **FastAPI integration** (`app/main.py`)  
   - `POST /generate`: (1) calls `generate_questions`, (2) runs `filter_valid_items` on the returned items, (3) returns only items predicted as valid.  
   - Response includes `n_requested` and `n_returned` so filtering is visible.

6. **Dataset**  
   - Default dataset: `data/all_questions_deduplicated.csv` (in-repo).  
   - Override with `BLOOM_DATASET_PATH` in `.env` if using a different file.

7. **Evaluation**  
   - Training script prints validation and test metrics and writes them to `models/quality_classifier_metrics.json` (including `split_pct`).

8. **Tests**  
   - `tests/test_quality_classifier.py`: unit tests for `predict` and `filter_valid_items` (with mocked model).

9. **Dependencies** (`requirements.txt`)  
   - Added: `pandas`, `scikit-learn`, `joblib` for the classifier; existing FastAPI, OpenAI, etc. for the API.

### How Everything Is Orchestrated

```
┌─────────────────────────────────────────────────────────────────────────┐
│ 1. TRAINING (one-time or when retraining)                               │
│    data/all_questions_deduplicated.csv                                 │
│         → scripts/train_quality_classifier.py                           │
│         → 70% train / 15% val / 15% test (stratified)                  │
│         → TF-IDF + LogisticRegression                                  │
│         → models/quality_classifier.joblib + quality_classifier_metrics │
└─────────────────────────────────────────────────────────────────────────┘

┌─────────────────────────────────────────────────────────────────────────┐
│ 2. API REQUEST (POST /generate)                                        │
│    Request: { topic, bloom, qtype, n }                                  │
│         → app.llm.generator.generate_questions()  (Azure OpenAI)        │
│         → list of items (question, choices, correct_index, …)            │
│         → app.quality.classifier.filter_valid_items(items)               │
│         → only items with predicted label 1 kept                         │
│    Response: { topic, target_bloom, type, n_requested, n_returned, items }│
└─────────────────────────────────────────────────────────────────────────┘
```

- **Training** is separate from the API: run the script to produce/update the model; the API only loads the saved artifact.  
- **Inference** is a pure filter: no Bloom labels are predicted; we only predict binary quality (0/1) and drop 0s.

### Results (from a full training run)

- **Data**: `data/all_questions_deduplicated.csv` (labeled questions; schema preserved).
- **Split**: Train 70%, Validation 15%, Test 15% (stratified).

| Set        | Accuracy | Precision | Recall | F1    |
|-----------|----------|-----------|--------|-------|
| Validation | 0.7388  | 0.7288    | 0.8683 | 0.7925 |
| Test       | **0.7589** | 0.7522  | 0.8639 | 0.8042 |

- **Test accuracy**: **75.89%** (accuracy_score on the held-out test set).
- **Artifacts**: `models/quality_classifier.joblib`, `models/quality_classifier_metrics.json` (contains `test`, `validation`, and `split_pct`).

So the system **learns** from the dataset (supervised binary classifier), is **integrated** into the pipeline (filter after LLM), and results are **documented** and **testable** (metrics + tests).

---

## Setup

### 1. Environment

Copy `.env.example` to `.env` and set:

- `AZURE_OPENAI_API_KEY`, `AZURE_OPENAI_ENDPOINT`, `AZURE_OPENAI_DEPLOYMENT`, `AZURE_OPENAI_API_VERSION` (for LLM generation).
- Optional: `BLOOM_DATASET_PATH` — path to your labeled CSV (default: `data/all_questions_deduplicated.csv`).  
- Optional: `BLOOM_QUALITY_MODEL_PATH` — path to saved model (default: `models/quality_classifier.joblib`).

### 2. Install dependencies

```bash
pip install -r requirements.txt
```

### 3. Train the classifier

Uses the dataset at `data/all_questions_deduplicated.csv` (or `BLOOM_DATASET_PATH`). Produces the model and metrics.

```bash
python scripts/train_quality_classifier.py
```

Output: `models/quality_classifier.joblib`, `models/quality_classifier_metrics.json`.

### 4. Run the API

```bash
python -m uvicorn app.main:app --reload
```
(Or `uvicorn app.main:app --reload` if uvicorn is on your PATH.)

- `GET /health` — service and Azure config.
- `POST /generate` — generate questions (Bloom in prompt), then filter by the classifier; response includes `n_requested` and `n_returned`.

---

## Code Layout

| Piece | Path | Role |
|-------|------|------|
| Config | `app/config.py` | `DATASET_PATH`, `QUALITY_MODEL_PATH` (env overrides). |
| Training | `scripts/train_quality_classifier.py` | Load CSV → train/val/test split → train pipeline → save model + metrics. |
| Inference | `app/quality/classifier.py` | Load model, `predict(question)`, `filter_valid_items(items)`. |
| LLM | `app/llm/generator.py` | `generate_questions(topic, bloom, qtype, n)` via Azure OpenAI. |
| API | `app/main.py` | `/generate`: generator → classifier filter → response. |
| Dataset | `data/all_questions_deduplicated.csv` | Columns: `question`, `label`, `is_labeled`, `source`, `dataset_type`. |
| Tests | `tests/test_quality_classifier.py` | Classifier predict and filter (mocked model). |

---

## Dataset

- **Schema**: `question`, `label`, `is_labeled`, `source`, `dataset_type` (all columns kept).
- **Label**: `1` = valid/content-related, `0` = invalid/meta/vague. Not Bloom-level labels.
- **Default path**: `data/all_questions_deduplicated.csv`. Override with `BLOOM_DATASET_PATH`.
