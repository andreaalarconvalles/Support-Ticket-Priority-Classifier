# 🎫 Support Ticket Priority Classifier
### IE University — Deep Learning Final Project · June 2026

---

## Project Overview

A Bidirectional LSTM model that reads incoming customer support ticket text and classifies it into priority tiers (**Low / Medium / High / Critical**), routing it automatically to the correct department. The frontend is a Streamlit web app any non-technical user can operate in seconds.

| | |
|---|---|
| **Architecture** | Bidirectional LSTM + Keras Embedding Layer (Current Accuracy: 65%) |
| **Dataset** | Support Ticket Priority Dataset — 50K rows |
| **Dataset URL** | kaggle.com/datasets/albertobircoci/support-ticket-priority-dataset-50k |
| **Backend** | FastAPI + saved Keras model (`models/ticket_classifier_tuned.keras`) |
| **Frontend** | Streamlit (Python) — Customer Support Email interface (Subject + Body) |
| **Team** | 3 people · 4-day sprint |

---

## Team Roles

### 🧠 Juan José — ML Engineer
> Owns the full machine learning pipeline end to end.

- Text preprocessing pipeline (clean, tokenise, pad, encode labels)
- Keras Embedding layer setup
- Bi-LSTM model architecture
- Training, evaluation, and tuning (accuracy, Macro F1, confusion matrix)
- **Handoff:** `ticket_classifier.keras` + `tokenizer.pkl` → Person 2 by Day 3

---

### ⚙️ Person 2 — Data & Backend Engineer
> Owns data preparation and the API that connects model to frontend.

- GitHub repo setup + dataset download
- EDA: class distribution chart, text length histogram, label quality check
- `preprocessing.py` — reusable pipeline handed to Juan José by Day 1 EOD
- FastAPI `/predict` endpoint (mock first, real model on Day 3)
- **Handoff:** Preprocessed arrays + tokenizer → Juan José (Day 1) · API URL → Person 3 (Day 2)

---

### 🎨 Person 3 — Frontend & Presentation Lead
> Owns the Streamlit app, the presentation deck, and the live demo.

- Streamlit app with ticket input, priority badge, confidence bars, demo buttons
- Connect frontend to real API on Day 3
- Full presentation deck (9 slides + demo script)
- Rehearsal, timing, speaking assignments, backup screenshots
- **Handoff:** Working app + clean GitHub README by Day 4 morning

---

## 4-Day Timeline

```
              JUAN JOSÉ (ML)              PERSON 2 (Data + Backend)      PERSON 3 (Frontend + Deck)
─────────────────────────────────────────────────────────────────────────────────────────────────────
DAY 1   Study Bi-LSTM architecture        Download 50K dataset            Set up GitHub repo
        Set up Colab environment          Run EDA (class dist,            Build Streamlit shell
        Agree JSON contract               text lengths, label check)      with hardcoded mock output
        Prepare training notebook         Build preprocessing.py          Draft presentation outline
                                          Save tokenizer.pkl + arrays     Write business problem slides
                                          ── HANDOFF TO JUAN JOSÉ ──>

DAY 2   Load preprocessed arrays          Build FastAPI /predict          Connect Streamlit to mock API
        Build Bi-LSTM model               endpoint (MOCK_MODE=True)       Add 4 demo ticket buttons
        Start training (run overnight)    Test with curl/Postman          Polish priority badge UI
                                          ── SHARE API URL ──>            Draft architecture slides

DAY 3   Evaluate model                    Receive model + tokenizer       Swap mock for real API
        Confusion matrix + F1             Set MOCK_MODE=False             Test full end-to-end flow
        Tune if needed                    Test all 4 demo tickets         Fill in metric slides
        ── HANDOFF MODEL ──>              Fix any integration issues      Finalise presentation deck

DAY 4   Integration testing               Fix any remaining API issues    Full rehearsal × 2 (timed)
        Q&A prep                          Confirm GitHub complete         Demo script practice
        Support teammates                 Q&A prep                        Confirm all speakers ready
```

---

## Critical Handoffs

| When | From | To | Deliver |
|---|---|---|---|
| Day 1 EOD | Person 2 | Juan José | `X_train/val/test.npy` · `y_train/val/test.npy` · `tokenizer.pkl` · note with `VOCAB_SIZE`, `MAX_SEQ_LEN`, `NUM_CLASSES` |
| Day 2 | Person 2 | Person 3 | FastAPI running on `localhost:8000` · curl test confirming JSON shape |
| Day 3 AM | Juan José | Person 2 | `ticket_classifier.keras` · confirm `tokenizer.pkl` unchanged · F1 + accuracy numbers |
| Day 3 PM | Person 2 | Person 3 | Confirm `MOCK_MODE=False` works · all 4 demo tickets return correct priority |

---

## API Contract — Agree Day 1, Never Change

Everyone builds against this exact shape. Person 2 produces it. Person 3 consumes it.

```json
{
  "priority": "Critical",
  "confidence": {
    "Low": 0.02,
    "Medium": 0.05,
    "High": 0.11,
    "Critical": 0.82
  },
  "department": "Incident Response"
}
```

---

## GitHub Structure

```
/
├── backend/
│   ├── data/                        # Raw CSV datasets (gitignore if large)
│   ├── models/                      # ticket_classifier.keras + tokenizer.pkl
│   ├── preprocessing.py             # Reusable text cleaning + padding pipeline
│   ├── eda.ipynb                    # EDA notebook with charts
│   ├── model_training.ipynb         # Full training notebook with outputs
│   ├── api.py                       # FastAPI /predict endpoint
│   └── requirements.txt            # tensorflow, fastapi, uvicorn, sklearn, nltk
├── frontend/
│   ├── app.py                       # Streamlit application
│   └── requirements.txt            # streamlit, requests, pandas
└── README.md                        # Setup instructions + model metrics
```

---

## Deliverables Checklist (Status Update)

**✅ Completed (Model Phase):**
- **Data Pipeline:** `preprocessing.py` + numpy arrays + `tokenizer.pkl`
- **EDA:** `eda.ipynb` with class distribution + length charts
- **Model Training:** `ticket_classifier_tuned.keras` trained model (Achieved **65% Accuracy**)
- **Metrics:** Accuracy, Macro F1, confusion matrix image generated.

**🚀 Missing & Next Steps (Production & Integration Phase):**
1. **FastAPI Backend (The API Bridge)**
   - Need to build the `/predict` endpoint that loads `models/ticket_classifier_tuned.keras`.
   - The API is critical because it separates our heavy ML model from the lightweight frontend, exposing a simple prediction service.
2. **Streamlit Frontend (The UI)**
   - Build a UI tailored to our dataset format: representing a Customer Support Email.
   - It needs to capture the **Subject** and **Body** of incoming tickets to feed into our API.
3. **Presentation & Cleanup**
   - Full presentation deck (PDF).
   - Clean `README.md` with final demo instructions.

---

## Presentation Structure (15 minutes)

| # | Slide | Speaker | Time |
|---|---|---|---|
| 1 | Title — project name, team, IE University | All | 30s |
| 2 | Business Problem — triage cost, SLA breaches, the pain | Person 3 | 2 min |
| 3 | Solution — Bi-LSTM flow diagram | Person 3 | 1 min |
| 4 | Architecture — layers, why Bidirectional, embedding approach | Juan José | 2 min |
| 5 | Dataset — 50K tickets, 4 classes, class distribution chart | Person 2 | 1 min |
| 6 | Results — Accuracy, Macro F1, Confusion Matrix | Juan José | 2 min |
| 7 | Live Demo — Streamlit, run all 4 demo tickets in order | Person 3 | 4 min |
| 8 | Business Impact — ROI, time saved, SLA compliance | Person 3 | 1 min |
| 9 | Q&A | All | 2 min |

---

## Grading Rubric

| Criterion | Weight | How to score full marks |
|---|---|---|
| Business Use Case & Value Proposition | 20% | Quantify ROI: 1,000 tickets/day × 3 min saved = 50 hrs/day. Clear SLA breach cost framing. |
| Technical Depth & Model Architecture | 25% | Justify Bi-LSTM. Show confusion matrix. Report Macro F1 not just accuracy. Show dropout + early stopping. |
| MVP Integration & Frontend UX | 25% | Seamless end-to-end demo. UI is intuitive without explanation. Priority badge is instant and clear. |
| Presentation & Team Delivery | 20% | All 3 members speak. Smooth transitions. No reading from notes. Clean slides. |
| Live Demo & Time Management | 10% | Demo works live. Finish within 15 min. Have backup screenshots ready just in case. |

---

*Individual technical guides available separately for each team member.*
