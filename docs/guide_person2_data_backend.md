# 🎫 Deep Learning Project — Your Guide
## Role: Data & Backend Engineer
**Project:** Support Ticket Priority Classifier | IE University Deep Learning Final Project

---

> **Your job in one sentence:** Get the data ready for Juan José to train the model, then turn his trained model into a working API that the frontend can call.

---

## ⚠️ Read This First

You have two distinct phases with a waiting window in between:

- **Day 1:** Data, EDA, and preprocessing → hand off to Juan José by end of day
- **Day 2 (while waiting for model):** Build the FastAPI skeleton against mock output
- **Day 3:** Swap mock for real model, test end-to-end

Do not skip the Day 1 handoff. Juan José cannot start building until he has your output.

---

## 📦 What You Need to Install

```bash
pip install pandas numpy matplotlib seaborn scikit-learn
pip install tensorflow keras nltk
pip install fastapi uvicorn python-multipart
pip install kaggle
```

Set up a Kaggle account if you don't have one — you need it to download the datasets. Go to kaggle.com → Account → Create API Token → download `kaggle.json` and place it at `~/.kaggle/kaggle.json`.

---

---

# DAY 1 — Data, EDA & Preprocessing (✅ COMPLETED)

## Step 1 — Download the Datasets (Done)

The dataset used is the **Customer Support Ticket Dataset (8,469 rows)**:
https://www.kaggle.com/datasets/muqaddasejaz/customer-support-ticket-dataset

*Note: The original 50K dataset was replaced with this one as it fits the schema perfectly.*

---

## Step 2 — Exploratory Data Analysis (EDA)

Open a Jupyter notebook called `eda.ipynb`. Run the following checks and save the output charts — they go directly into the presentation slides.

### 2a. Load and inspect the data

```python
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns

df = pd.read_csv("data/support_ticket_priority_50k.csv")
print(df.shape)
print(df.columns.tolist())
print(df.head(3))
print(df.dtypes)
print(df.isnull().sum())
```

Identify which columns contain: the ticket text, the priority label. Column names may vary — find them by inspection.

### 2b. Class distribution — MOST IMPORTANT CHART

```python
plt.figure(figsize=(8, 5))
df['priority'].value_counts().plot(kind='bar', color=['#10B981','#F59E0B','#EF4444','#7C3AED'])
plt.title('Ticket Priority Distribution')
plt.xlabel('Priority Class')
plt.ylabel('Count')
plt.xticks(rotation=0)
plt.tight_layout()
plt.savefig('charts/class_distribution.png')
plt.show()
```

**What to look for:** If one class has fewer than 10% of the samples, flag it for Juan José. He'll need to use class weights during training.

### 2c. Text length distribution

```python
import nltk
nltk.download('punkt')

# Combine subject + body if both columns exist
df['full_text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')

df['token_count'] = df['full_text'].apply(lambda x: len(x.split()))

plt.figure(figsize=(10, 5))
df['token_count'].hist(bins=50, color='#3B82F6', edgecolor='white')
plt.axvline(df['token_count'].quantile(0.95), color='red', linestyle='--', label='95th percentile')
plt.title('Ticket Text Length Distribution (in tokens)')
plt.xlabel('Number of tokens')
plt.ylabel('Count')
plt.legend()
plt.tight_layout()
plt.savefig('charts/text_length.png')
plt.show()

print(f"95th percentile token count: {df['token_count'].quantile(0.95):.0f}")
print(f"Mean token count: {df['token_count'].mean():.0f}")
print(f"Max token count: {df['token_count'].max():.0f}")
```

**Write down the 95th percentile number** — this becomes the `MAX_SEQUENCE_LENGTH` you pass to Juan José.

### 2d. Manual label quality check

Manually read 5–10 tickets from each priority class. Just print them:

```python
for priority in df['priority'].unique():
    print(f"\n{'='*50}")
    print(f"PRIORITY: {priority}")
    print('='*50)
    sample = df[df['priority'] == priority]['full_text'].head(3).values
    for i, text in enumerate(sample):
        print(f"\n[{i+1}] {text[:300]}")
```

Ask yourself: does the language feel different between classes? Note anything unusual for the team.

---

## Step 3 — Preprocessing Pipeline (Done by Juan José)

This has been completed and the output is ready in `data/processed/`. You can skip to Day 2, but here is the reference code that was used:

```python
# backend/preprocessing.py

import pandas as pd
import numpy as np
import re
import pickle
from sklearn.model_selection import train_test_split
from sklearn.preprocessing import LabelEncoder
from tensorflow.keras.preprocessing.text import Tokenizer
from tensorflow.keras.preprocessing.sequence import pad_sequences
from tensorflow.keras.utils import to_categorical

# ─── CONFIGURATION ───────────────────────────────────────────────
MAX_SEQUENCE_LENGTH = 150   # Replace with your 95th percentile number from EDA
MAX_VOCAB_SIZE = 15000
TEST_SIZE = 0.15
VAL_SIZE = 0.15
RANDOM_STATE = 42
# ─────────────────────────────────────────────────────────────────

def clean_text(text: str) -> str:
    """Clean a single text string."""
    if not isinstance(text, str):
        return ""
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)        # remove URLs
    text = re.sub(r'\S+@\S+', '', text)                # remove emails
    text = re.sub(r'[^a-z0-9\s!?]', '', text)         # keep letters, nums, !?
    text = re.sub(r'\s+', ' ', text).strip()           # collapse whitespace
    return text

def load_and_prepare(csv_path: str):
    """Load CSV and return cleaned text + labels."""
    df = pd.read_csv(csv_path)
    
    # ── Adjust these column names to match your actual dataset ──
    df['full_text'] = df['subject'].fillna('') + ' ' + df['body'].fillna('')
    df['full_text'] = df['full_text'].apply(clean_text)
    df = df[df['full_text'].str.len() > 10].reset_index(drop=True)
    
    return df['full_text'].values, df['priority'].values

def build_and_save_tokenizer(texts_train, save_path='models/tokenizer.pkl'):
    """Fit tokenizer on training data only and save it."""
    tokenizer = Tokenizer(num_words=MAX_VOCAB_SIZE, oov_token='<OOV>')
    tokenizer.fit_on_texts(texts_train)
    with open(save_path, 'wb') as f:
        pickle.dump(tokenizer, f)
    print(f"Tokenizer saved to {save_path}")
    print(f"Vocabulary size: {len(tokenizer.word_index)}")
    return tokenizer

def encode_and_pad(texts, tokenizer):
    """Convert texts to padded sequences."""
    sequences = tokenizer.texts_to_sequences(texts)
    padded = pad_sequences(sequences, maxlen=MAX_SEQUENCE_LENGTH,
                           padding='post', truncating='post')
    return padded

def encode_labels(labels):
    """Encode string labels to one-hot vectors."""
    le = LabelEncoder()
    labels_int = le.fit_transform(labels)
    labels_onehot = to_categorical(labels_int)
    return labels_onehot, le.classes_

def run_full_pipeline(csv_path: str, save_dir: str = 'models/'):
    """Run the full preprocessing pipeline and save outputs."""
    import os
    os.makedirs(save_dir, exist_ok=True)
    os.makedirs('charts', exist_ok=True)

    print("Loading data...")
    texts, labels = load_and_prepare(csv_path)
    
    print("Splitting data...")
    X_temp, X_test, y_temp, y_test = train_test_split(
        texts, labels, test_size=TEST_SIZE, random_state=RANDOM_STATE, stratify=labels)
    X_train, X_val, y_train, y_val = train_test_split(
        X_temp, y_temp, test_size=VAL_SIZE/(1-TEST_SIZE),
        random_state=RANDOM_STATE, stratify=y_temp)
    
    print(f"Train: {len(X_train)} | Val: {len(X_val)} | Test: {len(X_test)}")
    
    print("Building tokenizer...")
    tokenizer = build_and_save_tokenizer(X_train, save_path=f'{save_dir}tokenizer.pkl')
    
    print("Encoding and padding sequences...")
    X_train_pad = encode_and_pad(X_train, tokenizer)
    X_val_pad   = encode_and_pad(X_val,   tokenizer)
    X_test_pad  = encode_and_pad(X_test,  tokenizer)
    
    print("Encoding labels...")
    y_train_enc, class_names = encode_labels(y_train)
    y_val_enc,   _           = encode_labels(y_val)
    y_test_enc,  _           = encode_labels(y_test)
    
    print("Saving arrays...")
    np.save(f'{save_dir}X_train.npy', X_train_pad)
    np.save(f'{save_dir}X_val.npy',   X_val_pad)
    np.save(f'{save_dir}X_test.npy',  X_test_pad)
    np.save(f'{save_dir}y_train.npy', y_train_enc)
    np.save(f'{save_dir}y_val.npy',   y_val_enc)
    np.save(f'{save_dir}y_test.npy',  y_test_enc)
    
    print("\n✅ Pipeline complete. Summary for Juan José:")
    print(f"   VOCAB_SIZE         = {min(MAX_VOCAB_SIZE, len(tokenizer.word_index))}")
    print(f"   MAX_SEQUENCE_LENGTH = {MAX_SEQUENCE_LENGTH}")
    print(f"   NUM_CLASSES        = {len(class_names)}")
    print(f"   CLASS NAMES        = {list(class_names)}")
    
    return X_train_pad, X_val_pad, X_test_pad, y_train_enc, y_val_enc, y_test_enc, class_names

if __name__ == '__main__':
    run_full_pipeline('data/support_ticket_priority_50k.csv')
```

Run it. Fix any column name errors. When it completes successfully, commit everything to GitHub.

---

## ✅ End of Day 1 Handoff to Juan José

Message Juan José with exactly these four numbers:

```
VOCAB_SIZE          = [number from output]
MAX_SEQUENCE_LENGTH = [number from your EDA]
NUM_CLASSES         = [3 or 4 depending on dataset]
CLASS_NAMES         = [e.g. ['Critical', 'High', 'Low', 'Medium']]
```

And confirm that these files are committed to `/backend/models/`:
- `tokenizer.pkl`
- `X_train.npy`, `X_val.npy`, `X_test.npy`
- `y_train.npy`, `y_val.npy`, `y_test.npy`

---

---

# DAY 2 — Build the FastAPI Backend (with mock model)

While Juan José trains the model, build the full API structure against a mock prediction function. This way you are not blocked.

Create `/backend/api.py`:

```python
# backend/api.py

import pickle
import numpy as np
from fastapi import FastAPI
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel
import re

# ── Will be replaced with real model on Day 3 ──
MOCK_MODE = True

app = FastAPI(title="Support Ticket Priority Classifier")

app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_methods=["*"],
    allow_headers=["*"],
)

# ── Load assets at startup ──
model = None
tokenizer = None

MAX_SEQUENCE_LENGTH = 150  # Match what Person 2 used in preprocessing

CLASS_NAMES = ['Low', 'Medium', 'High', 'Critical']
DEPARTMENT_MAP = {
    'Low':      'General Support',
    'Medium':   'Customer Service',
    'High':     'Technical Support',
    'Critical': 'Incident Response'
}

@app.on_event("startup")
def load_model():
    global model, tokenizer
    if not MOCK_MODE:
        from tensorflow.keras.models import load_model as keras_load
        model = keras_load("models/ticket_classifier_tuned.keras")
        with open("models/tokenizer.pkl", "rb") as f:
            tokenizer = pickle.load(f)
        print("✅ Model and tokenizer loaded")

# ── Request / Response schemas ──
class TicketRequest(BaseModel):
    subject: str = ""
    body: str = ""

class TicketResponse(BaseModel):
    priority: str
    confidence: dict
    department: str

# ── Preprocessing (mirrors Person 2's pipeline) ──
def clean_text(text: str) -> str:
    text = text.lower()
    text = re.sub(r'http\S+|www\S+', '', text)
    text = re.sub(r'\S+@\S+', '', text)
    text = re.sub(r'[^a-z0-9\s!?]', '', text)
    text = re.sub(r'\s+', ' ', text).strip()
    return text

def preprocess(subject: str, body: str):
    from tensorflow.keras.preprocessing.sequence import pad_sequences
    combined = clean_text(subject + ' ' + body)
    seq = tokenizer.texts_to_sequences([combined])
    padded = pad_sequences(seq, maxlen=MAX_SEQUENCE_LENGTH,
                           padding='post', truncating='post')
    return padded

# ── Prediction endpoint ──
@app.post("/predict", response_model=TicketResponse)
def predict(ticket: TicketRequest):
    if MOCK_MODE:
        # Hardcoded mock — remove on Day 3
        probs = [0.05, 0.10, 0.20, 0.65]
    else:
        padded = preprocess(ticket.subject, ticket.body)
        probs = model.predict(padded)[0].tolist()
    
    predicted_idx = int(np.argmax(probs))
    predicted_class = CLASS_NAMES[predicted_idx]
    
    return TicketResponse(
        priority=predicted_class,
        confidence={name: round(float(p), 4) for name, p in zip(CLASS_NAMES, probs)},
        department=DEPARTMENT_MAP[predicted_class]
    )

@app.get("/health")
def health():
    return {"status": "ok", "mock_mode": MOCK_MODE}
```

Run it locally:

```bash
uvicorn api:app --reload --port 8000
```

Test it:

```bash
curl -X POST http://localhost:8000/predict \
  -H "Content-Type: application/json" \
  -d '{"subject": "System is down", "body": "All users are affected"}'
```

You should see a JSON response. Share this with Person 3 so they can connect the frontend.

---

# DAY 3 — Swap Mock for Real Model

When Juan José hands over `ticket_classifier.keras` and confirms the tokenizer is the same one already saved:

1. Set `MOCK_MODE = False` in `api.py`
2. Restart the server
3. Run the same curl test — response should now reflect the real model
4. Run these 4 test tickets through Postman or curl and record the outputs for Person 3's demo script:

```python
test_tickets = [
    {"subject": "URGENT: Production database completely down",
     "body": "All users cannot access the platform. We are losing revenue every minute."},
    {"subject": "Login page throwing 500 errors intermittently",
     "body": "About 30% of users are reporting issues logging in over the past hour."},
    {"subject": "How do I export my data to CSV?",
     "body": "I need to download my account data for a report. Where can I find this?"},
    {"subject": "Change notification preferences",
     "body": "I'd like to stop receiving weekly summary emails. How do I do this?"}
]
```

---

## ✅ Your Final Deliverables

By end of Day 3, confirm these all work:

- [ ] `POST /predict` returns correct JSON structure
- [ ] `GET /health` returns 200
- [ ] API runs with `uvicorn api:app --port 8000` from `/backend/`
- [ ] All 4 demo tickets return sensible priority labels
- [ ] API is committed to GitHub with a comment in the README on how to run it

---

## 🚨 Common Issues and Fixes

**"Module not found: tensorflow"** → Run `pip install tensorflow` in your virtual environment, not globally.

**Tokenizer gives all OOV tokens** → Make sure you are loading `tokenizer.pkl` from the exact file Person 2 saved, not rebuilding it.

**CORS error from Streamlit** → The `CORSMiddleware` block in the API handles this. If it still fails, confirm you added it before any routes.

**Model prediction always returns the same class** → Check that labels were not accidentally all encoded to the same integer. Ask Juan José to confirm the class name order.
