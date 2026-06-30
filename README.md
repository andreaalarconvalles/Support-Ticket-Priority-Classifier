# Support Ticket Priority Classifier

IE University · Deep Learning A · Final Project (June 2026)

A deep learning MVP that automatically classifies incoming customer support tickets into priority tiers (**Low, Medium, High**) and routes them to the correct support department, so non-technical agents get an instant triage call instead of doing it by hand.

## Architecture

| | |
|---|---|
| Backend model | Bidirectional LSTM (Keras / TensorFlow), text → priority (Test Accuracy: 64.99%, Macro F1: 0.6435) |
| Department routing | Plain lookup from `Ticket Type` (see `src/utils/department_mapping.py`) — not a model output, since `Ticket Type` is already known per ticket |
| Backend serving | FastAPI `/predict` endpoint |
| Frontend | Streamlit app (Customer Support Email interface: Subject + Body) |
| Dataset | [Multilingual Customer Support Tickets](https://www.kaggle.com/datasets/tobiasbueck/multilingual-customer-support-tickets) (Kaggle) — 28,587 tickets (EN/DE), filtered to **16,338 English tickets** across 3 priority classes |

> Note: the team switched datasets again after the original 8,469-ticket, 4-class version — this new dataset is bilingual and larger, but only has 3 priority classes (no "Critical") and is class-imbalanced rather than balanced.

## Key data findings (from EDA)

- Raw dataset is bilingual (English/German); filtered to **English only** → 16,338 of 28,587 rows.
- Priority classes are **imbalanced**: Low 21% (3,374), Medium 41% (6,618), High 39% (6,346) — `src/models/train_classifier.py` computes balanced class weights to compensate.
- `MAX_SEQ_LEN` is fixed at **57** (carried over from the prior dataset's config — not re-derived from this dataset's length distribution).
- No duplicate `full_text` (Subject + Body) rows remain after filtering to English — `preprocessing.py` still dedupes before splitting as a safety check.
- No `department` column exists in the new dataset's used fields; resolved as a `Ticket Type` → department lookup (see table below). Note: the raw CSV's `queue` column already contains department-like categories (e.g. "Technical Support", "Billing and Payments") that the current lookup doesn't use yet — worth revisiting.

See `notebooks/01_eda.ipynb` for the full analysis with charts.

## Test results

Final test accuracy: **0.6499** · Final test Macro F1: **0.6435**

| Class | Precision | Recall | F1-score | Support |
|---|---|---|---|---|
| Low | 0.65 | 0.58 | 0.61 | 506 |
| Medium | 0.64 | 0.66 | 0.65 | 993 |
| High | 0.66 | 0.68 | 0.67 | 952 |
| **Accuracy** | | | **0.65** | 2,451 |
| **Macro avg** | 0.65 | 0.64 | 0.64 | 2,451 |
| **Weighted avg** | 0.65 | 0.65 | 0.65 | 2,451 |

Recall is weakest on **Low** (0.58) — the model under-detects the minority class, consistent with it being the smallest of the three (21% of data) despite balanced class weights. High and Medium perform comparably, both above the overall accuracy.

## Team

| Person | Role |
|---|---|
| Juan José | ML Engineer — preprocessing pipeline, embeddings, Bi-LSTM model, training/evaluation |
| Andrea (Person 2) | Data & Backend Engineer — repo/env setup, EDA, preprocessing handoff, FastAPI backend |
| Person 3 | Frontend & Presentation Lead — Streamlit app, integration, final deck |

## Repo structure

```text
Support-Ticket-Priority-Classifier/
├── data/                   # Data directory
│   ├── raw/                # The original, immutable downloaded data
│   └── processed/          # Cleaned and processed data ready for training
├── models/                 # Saved final models (e.g., ticket_classifier_tuned.keras)
├── tuning_logs/            # Logs from hyperparameter tuning (e.g., Optuna, MLflow)
├── notebooks/              # Jupyter notebooks for EDA and playground
│   ├── 01_eda.ipynb        
│   └── 02_playground.ipynb 
├── src/                    # Source code for data processing and training pipelines
│   ├── data/               # Scripts to fetch and clean data
│   │   ├── download_data.py
│   │   └── preprocessing.py
│   ├── models/             # Scripts to train, tune, and evaluate models
│   │   ├── train_classifier.py
│   │   └── tune_classifier.py
│   └── utils/              # Helper functions and mappings
│       └── department_mapping.py
├── app/                    # Web Application code (coming soon)
│   ├── api/                # FastAPI backend connecting UI to model inference
│   └── frontend/           # UI code for the end user
├── docs/                   # Documentation
├── requirements.txt        # Project dependencies
└── README.md               # Top-level README
```

### Phase 3 handoff (for Juan José)

Run `python src/data/preprocessing.py` to regenerate `data/processed/`. Key numbers (also in `data/processed/HANDOFF_NOTES.txt`):

| | |
|---|---|
| `VOCAB_SIZE` | 5,726 |
| `MAX_SEQ_LEN` | 57 |
| `NUM_CLASSES` | 3 |
| `LABEL_TO_ID` | `{"low": 0, "medium": 1, "high": 2}` |
| Split | 70/15/15, stratified by `priority`, `random_state=42` (11,436 / 2,451 / 2,451) |

## Setup

```bash
git clone https://github.com/andreaalarconvalles/Support-Ticket-Priority-Classifier.git
cd Support-Ticket-Priority-Classifier

conda create -n ml python=3.10
conda activate ml
pip install -r requirements.txt
pip install -r app/frontend/requirements.txt

python src/data/download_data.py
```

## API contract

Fixed on Day 1 so backend and frontend can be built in parallel:

```json
{
  "priority": "High",
  "confidence": {"Low": 0.05, "Medium": 0.17, "High": 0.78},
  "department": "Technical Support"
}
```

`department` values come from this lookup (`src/utils/department_mapping.py`):

| Ticket Type | → Department |
|---|---|
| Technical issue | Technical Support |
| Billing inquiry | Billing |
| Refund request | Billing |
| Cancellation request | Customer Retention |
| Product inquiry | Sales |

## Status & Next Steps

### What's Done
- [x] Repo structure, environment, dataset download
- [x] EDA (class distribution, text length, label quality)
- [x] Department routing resolved (`Ticket Type` lookup, not a model output)
- [x] Preprocessing pipeline complete: clean, dedupe, split, tokenize, pad, encode labels
- [x] Bi-LSTM model training & evaluation: Achieved **64.99% test accuracy** (Macro F1: 0.6435). This is sufficient for MVP and unblocks the rest of the project pipeline.

### Model Production Usage
The final trained model is stored in the `models/` directory (e.g., `ticket_classifier_tuned.keras`). 
To use the model in production:
1. Load the model using Keras: `tf.keras.models.load_model('models/ticket_classifier_tuned.keras')`
2. Load the accompanying tokenizer (`tokenizer.pkl`) to preprocess incoming text identically to the training phase.
3. Pass the padded sequences to the model's `predict()` function to get the confidence scores for each priority class.

### What's Missing (Next 2 Big Steps)

Now that the core model is ready, we are focusing on the user-facing parts and integration:

1. **The API (Connecting Model to UI)**
   - [ ] Build a FastAPI backend with a `/predict` endpoint.
   - **Why:** The UI shouldn't run heavy ML models directly. The API acts as the bridge, receiving text from the frontend, asking the model for a prediction, and returning the priority JSON response.

2. **The Frontend UI (Customer Support Interface)**
   - [ ] Build a Streamlit application.
   - **Why:** We need a realistic interface for support agents. 
   - **Design:** The UI should be designed to accept standard customer support emails, specifically capturing the **Subject** and **Body** of the message to simulate a real inbox experience. This will then be sent to the API for classification.
   - [ ] Final presentation deck preparation.
