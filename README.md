# Support Ticket Priority Classifier

IE University · Deep Learning A · Final Project (June 2026)

A deep learning MVP that automatically classifies incoming customer support tickets into priority tiers (**Low, Medium, High, Critical**) and routes them to the correct support department, so non-technical agents get an instant triage call instead of doing it by hand.

## Architecture

| | |
|---|---|
| Backend model | Bidirectional LSTM (Keras / TensorFlow), text → priority + department |
| Backend serving | FastAPI `/predict` endpoint |
| Frontend | Streamlit app |
| Dataset | [Customer Support Ticket Dataset](https://www.kaggle.com/datasets/muqaddasejaz/customer-support-ticket-dataset) (Kaggle) — 8,469 tickets, balanced across 4 priority classes |

> Note: the original project brief referenced a different 50K-row Kaggle dataset. The team switched to the dataset above since it already matches the required schema (ticket text + 4-class priority) and is cleanly balanced.

## Team

| Person | Role |
|---|---|
| Juan José | ML Engineer — preprocessing pipeline, embeddings, Bi-LSTM model, training/evaluation |
| Andrea (Person 2) | Data & Backend Engineer — repo/env setup, EDA, preprocessing handoff, FastAPI backend |
| Person 3 | Frontend & Presentation Lead — Streamlit app, integration, final deck |

## Repo structure

```
backend/
  data/raw/              # downloaded dataset (customer_support_tickets.csv)
  download_data.py       # pulls dataset from Kaggle via kagglehub
  models/                # trained model + tokenizer land here (Git LFS)
  requirements.txt
frontend/
  requirements.txt
```

`backend/eda.ipynb`, `backend/preprocessing.py`, `backend/model_training.ipynb`, `backend/api.py`, and `frontend/app.py` are still to come (see Status below).

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
  "department": "Incident Response"
}
```

## Status

- [x] Repo structure, environment, dataset download
- [ ] EDA (class distribution, text length, label quality)
- [ ] Preprocessing pipeline (clean/tokenize/pad) + tokenizer handoff
- [ ] Bi-LSTM model training & evaluation (accuracy, Macro F1, confusion matrix)
- [ ] FastAPI `/predict` endpoint (mock → real model)
- [ ] Streamlit frontend + end-to-end integration
- [ ] Final presentation deck

Model metrics and a "how to run the full demo" section will be added here once training and integration are complete.
