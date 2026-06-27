# Support Ticket Priority Classifier

IE University · Deep Learning A · Final Project (June 2026)

A deep learning MVP that automatically classifies incoming customer support tickets into priority tiers (**Low, Medium, High, Critical**) and routes them to the correct support department, so non-technical agents get an instant triage call instead of doing it by hand.

## Architecture

| | |
|---|---|
| Backend model | Bidirectional LSTM (Keras / TensorFlow), text → priority (Current Accuracy: 65%) |
| Department routing | Plain lookup from `Ticket Type` (see `src/utils/department_mapping.py`) — not a model output, since `Ticket Type` is already known per ticket |
| Backend serving | FastAPI `/predict` endpoint |
| Frontend | Streamlit app (Customer Support Email interface: Subject + Body) |
| Dataset | [Customer Support Ticket Dataset](https://www.kaggle.com/datasets/muqaddasejaz/customer-support-ticket-dataset) (Kaggle) — 8,469 tickets, balanced across 4 priority classes |

> Note: the original project brief referenced a different 50K-row Kaggle dataset. The team switched to the dataset above since it already matches the required schema (ticket text + 4-class priority) and is cleanly balanced.

## Key data findings (from EDA)

- Priority classes are well balanced (~2,000–2,200 tickets each) — no resampling needed.
- 95th-percentile description length is **57 words** → use this for `MAX_SEQ_LEN`.
- **100% of descriptions** contain at least one unfilled `{placeholder}` token (this dataset is template-generated). `src/data/preprocessing.py` fills `product_purchased` with the real product name and strips everything else (no real value to substitute, including malformed/unmatched braces).
- ~4.6% exact-duplicate descriptions in the *raw* text, but most of those were rows where the literal `{product_purchased}` token matched across different products — once cleaned (real product name filled in), only 73 true duplicates remain (0.9%). `preprocessing.py` dedupes on the cleaned text before splitting, so no duplicate spans train/val/test.
- No `department` column exists; resolved as a `Ticket Type` → department lookup (see table below), not a second model output.

See `notebooks/01_eda.ipynb` for the full analysis with charts.

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
pip install -r requirements.txt
pip install -r app/frontend/requirements.txt

python src/data/download_data.py
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
- [x] Bi-LSTM model training & evaluation: Achieved an initial **65% accuracy**. This is sufficient for MVP and unblocks the rest of the project pipeline.

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
