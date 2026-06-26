# Support Ticket Priority Classifier

IE University · Deep Learning A · Final Project (June 2026)

A deep learning MVP that automatically classifies incoming customer support tickets into priority tiers (**Low, Medium, High, Critical**) and routes them to the correct support department, so non-technical agents get an instant triage call instead of doing it by hand.

## Architecture

| | |
|---|---|
| Backend model | Bidirectional LSTM (Keras / TensorFlow), text → priority |
| Department routing | Plain lookup from `Ticket Type` (see `backend/department_mapping.py`) — not a model output, since `Ticket Type` is already known per ticket |
| Backend serving | FastAPI `/predict` endpoint |
| Frontend | Streamlit app |
| Dataset | [Customer Support Ticket Dataset](https://www.kaggle.com/datasets/muqaddasejaz/customer-support-ticket-dataset) (Kaggle) — 8,469 tickets, balanced across 4 priority classes |

> Note: the original project brief referenced a different 50K-row Kaggle dataset. The team switched to the dataset above since it already matches the required schema (ticket text + 4-class priority) and is cleanly balanced.

## Key data findings (from EDA)

- Priority classes are well balanced (~2,000–2,200 tickets each) — no resampling needed.
- 95th-percentile description length is **57 words** → use this for `MAX_SEQ_LEN`.
- **100% of descriptions** contain at least one unfilled `{placeholder}` token (this dataset is template-generated). `backend/preprocessing.py` fills `product_purchased` with the real product name and strips everything else (no real value to substitute, including malformed/unmatched braces).
- ~4.6% exact-duplicate descriptions in the *raw* text, but most of those were rows where the literal `{product_purchased}` token matched across different products — once cleaned (real product name filled in), only 73 true duplicates remain (0.9%). `preprocessing.py` dedupes on the cleaned text before splitting, so no duplicate spans train/val/test.
- No `department` column exists; resolved as a `Ticket Type` → department lookup (see table below), not a second model output.

See `backend/eda.ipynb` for the full analysis with charts.

## Team

| Person | Role |
|---|---|
| Juan José | ML Engineer — preprocessing pipeline, embeddings, Bi-LSTM model, training/evaluation |
| Andrea (Person 2) | Data & Backend Engineer — repo/env setup, EDA, preprocessing handoff, FastAPI backend |
| Person 3 | Frontend & Presentation Lead — Streamlit app, integration, final deck |

## Repo structure

```
backend/
  data/raw/                  # downloaded dataset (customer_support_tickets.csv)
  data/processed/            # X/y train/val/test .npy + tokenizer.pkl (Phase 3 handoff)
  download_data.py           # pulls dataset from Kaggle via kagglehub
  eda.ipynb                  # class distribution, text length, label quality checks
  preprocessing.py           # clean -> dedupe -> split -> tokenize -> pad -> encode labels
  department_mapping.py      # Ticket Type -> department lookup
  models/                    # trained model lands here (Git LFS)
  requirements.txt
frontend/
  requirements.txt
```

`backend/model_training.ipynb`, `backend/api.py`, and `frontend/app.py` are still to come (see Status below).

### Phase 3 handoff (for Juan José)

Run `python backend/preprocessing.py` to regenerate `backend/data/processed/`. Key numbers (also in `backend/data/processed/HANDOFF_NOTES.txt`):

| | |
|---|---|
| `VOCAB_SIZE` | 5,968 |
| `MAX_SEQ_LEN` | 57 |
| `NUM_CLASSES` | 4 |
| `LABEL_TO_ID` | `{"Low": 0, "Medium": 1, "High": 2, "Critical": 3}` |
| Split | 70/15/15, stratified by `Ticket Priority`, `random_state=42` |

## Setup

```bash
git clone https://github.com/andreaalarconvalles/Support-Ticket-Priority-Classifier.git
cd Support-Ticket-Priority-Classifier

conda create -n ml python=3.10
conda activate ml
pip install -r backend/requirements.txt
pip install -r frontend/requirements.txt

python backend/download_data.py
```

## API contract

Fixed on Day 1 so backend and frontend can be built in parallel:

```json
{
  "priority": "Critical",
  "confidence": {"Low": 0.02, "Medium": 0.05, "High": 0.11, "Critical": 0.82},
  "department": "Technical Support"
}
```

`department` values come from this lookup (`backend/department_mapping.py`):

| Ticket Type | → Department |
|---|---|
| Technical issue | Technical Support |
| Billing inquiry | Billing |
| Refund request | Billing |
| Cancellation request | Customer Retention |
| Product inquiry | Sales |

## Status

- [x] Repo structure, environment, dataset download
- [x] EDA (class distribution, text length, label quality)
- [x] Department routing resolved (`Ticket Type` lookup, not a model output)
- [x] Preprocessing pipeline complete: clean, dedupe, split, tokenize, pad, encode labels — handoff artifacts in `backend/data/processed/`
- [ ] Bi-LSTM model training & evaluation (accuracy, Macro F1, confusion matrix)
- [ ] FastAPI `/predict` endpoint (mock → real model)
- [ ] Streamlit frontend + end-to-end integration
- [ ] Final presentation deck

Model metrics and a "how to run the full demo" section will be added here once training and integration are complete.
