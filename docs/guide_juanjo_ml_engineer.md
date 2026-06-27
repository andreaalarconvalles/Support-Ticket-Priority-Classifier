# 🧠 Your Guide — ML Engineer
## Juan José · Support Ticket Priority Classifier

---

> **Your job in one sentence:** Take the preprocessed data from Person 2, build and train a Bi-LSTM model that classifies ticket priority, evaluate it properly, and hand the saved model + tokenizer to Person 2 by Day 3 morning.

---

## What You Receive from Person 2 (End of Day 1)

Before you write a single line of model code, confirm you have these files in `/backend/models/`:

```
X_train.npy        # Padded training sequences
X_val.npy          # Padded validation sequences
X_test.npy         # Padded test sequences
y_train.npy        # One-hot encoded training labels
y_val.npy          # One-hot encoded validation labels
y_test.npy         # One-hot encoded test labels
tokenizer.pkl      # Fitted Keras tokenizer — never refit this
```

And these four numbers in a message from Person 2:

```
VOCAB_SIZE          = ????
MAX_SEQUENCE_LENGTH = ????   (95th percentile of token counts from EDA)
NUM_CLASSES         = 4      (Low / Medium / High / Critical)
CLASS_NAMES         = ['Critical', 'High', 'Low', 'Medium']  (check exact order)
```

**Do not start training until you have all of this.** If Person 2 is delayed, use Day 1 to study the Bi-LSTM architecture and set up your Colab environment.

---

## Environment Setup

Use **Google Colab with a T4 GPU** — go to Runtime → Change runtime type → T4 GPU.

```python
# Install in Colab if needed
!pip install tensorflow scikit-learn matplotlib seaborn
```

Mount Google Drive to persist your model file between sessions:

```python
from google.colab import drive
drive.mount('/content/drive')
SAVE_DIR = '/content/drive/MyDrive/dl_project/'
import os; os.makedirs(SAVE_DIR, exist_ok=True)
```

---

## DAY 2 — Load Data & Build the Model

### Step 1 — Load Everything

```python
import numpy as np
import pickle

# Load preprocessed arrays
X_train = np.load('models/X_train.npy')
X_val   = np.load('models/X_val.npy')
X_test  = np.load('models/X_test.npy')
y_train = np.load('models/y_train.npy')
y_val   = np.load('models/y_val.npy')
y_test  = np.load('models/y_test.npy')

# Load tokenizer
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer = pickle.load(f)

print(f"X_train shape: {X_train.shape}")   # (n_samples, MAX_SEQ_LEN)
print(f"y_train shape: {y_train.shape}")   # (n_samples, NUM_CLASSES)
print(f"Vocab size: {len(tokenizer.word_index)}")
```

Sanity check: `X_train.shape[1]` should equal `MAX_SEQUENCE_LENGTH`. `y_train.shape[1]` should equal `NUM_CLASSES`.

---

### Step 2 — Check Class Imbalance

Before building the model, check whether you need class weights:

```python
import numpy as np

# y_train is one-hot encoded — argmax gives the class index
class_counts = np.sum(y_train, axis=0)
print("Samples per class:", class_counts)
print("Class percentages:", class_counts / class_counts.sum() * 100)
```

If any class has fewer than 15% of samples, compute class weights:

```python
from sklearn.utils.class_weight import compute_class_weight

y_int = np.argmax(y_train, axis=1)  # convert one-hot back to integers
classes = np.unique(y_int)
weights = compute_class_weight('balanced', classes=classes, y=y_int)
class_weight_dict = dict(zip(classes, weights))
print("Class weights:", class_weight_dict)
```

Pass `class_weight=class_weight_dict` to `model.fit()` later if needed.

---

### Step 3 — Build the Bi-LSTM Model

```python
import tensorflow as tf
from tensorflow.keras.models import Model
from tensorflow.keras.layers import (
    Input, Embedding, Bidirectional, LSTM,
    Dense, Dropout, GlobalMaxPooling1D
)

# ── Configuration — plug in numbers from Person 2 ──────────────────
VOCAB_SIZE          = 15000   # replace with actual
MAX_SEQUENCE_LENGTH = 150     # replace with actual
NUM_CLASSES         = 4
EMBEDDING_DIM       = 64
LSTM_UNITS          = 128
# ───────────────────────────────────────────────────────────────────

def build_model():
    inputs = Input(shape=(MAX_SEQUENCE_LENGTH,), name='token_input')

    x = Embedding(
        input_dim=VOCAB_SIZE + 1,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_SEQUENCE_LENGTH,
        trainable=True,
        name='embedding'
    )(inputs)

    x = Bidirectional(
        LSTM(LSTM_UNITS, return_sequences=False, dropout=0.2, recurrent_dropout=0.1),
        name='bi_lstm'
    )(x)

    x = Dropout(0.4, name='dropout_1')(x)
    x = Dense(64, activation='relu', name='dense_1')(x)
    x = Dropout(0.3, name='dropout_2')(x)

    outputs = Dense(NUM_CLASSES, activation='softmax', name='output')(x)

    model = Model(inputs=inputs, outputs=outputs)
    return model

model = build_model()
model.summary()
```

**Why Bidirectional?** The LSTM reads the ticket text both forward and backward. "System is completely down" carries different weight depending on context before and after it. This is your justification for the architecture choice — say it clearly in the presentation.

---

### Step 4 — Compile

```python
model.compile(
    optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
    loss='categorical_crossentropy',
    metrics=['accuracy']
)
```

---

## DAY 2-3 — Train, Monitor, Save

### Step 5 — Set Up Callbacks

```python
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau

callbacks = [
    EarlyStopping(
        monitor='val_loss',
        patience=5,
        restore_best_weights=True,
        verbose=1
    ),
    ModelCheckpoint(
        filepath=SAVE_DIR + 'ticket_classifier.keras',
        monitor='val_loss',
        save_best_only=True,
        verbose=1
    ),
    ReduceLROnPlateau(
        monitor='val_loss',
        factor=0.5,
        patience=3,
        verbose=1
    )
]
```

`EarlyStopping` stops training when validation loss stops improving — prevents overfitting. `ModelCheckpoint` saves only the best epoch, not the final one. These two together are the answer to any question about overfitting prevention in your Q&A.

---

### Step 6 — Train

```python
history = model.fit(
    X_train, y_train,
    validation_data=(X_val, y_val),
    epochs=30,
    batch_size=64,
    callbacks=callbacks,
    class_weight=class_weight_dict,  # remove this line if class balance was fine
    verbose=1
)
```

Training will take **10–20 minutes on a T4 GPU**. Start it before you go to sleep on Day 2 — Colab will run for a few hours unattended.

---

### Step 7 — Plot Training Curves

Run this immediately after training. Screenshot it — it goes into your presentation.

```python
import matplotlib.pyplot as plt

fig, (ax1, ax2) = plt.subplots(1, 2, figsize=(12, 4))

ax1.plot(history.history['loss'], label='Train Loss', color='#2563EB')
ax1.plot(history.history['val_loss'], label='Val Loss', color='#DC2626')
ax1.set_title('Loss over epochs')
ax1.set_xlabel('Epoch')
ax1.legend()

ax2.plot(history.history['accuracy'], label='Train Accuracy', color='#2563EB')
ax2.plot(history.history['val_accuracy'], label='Val Accuracy', color='#DC2626')
ax2.set_title('Accuracy over epochs')
ax2.set_xlabel('Epoch')
ax2.legend()

plt.tight_layout()
plt.savefig(SAVE_DIR + 'training_curves.png', dpi=150)
plt.show()
```

**What healthy curves look like:** both train and val loss decrease together, then val loss flattens and early stopping kicks in. If val loss starts rising while train loss keeps dropping, that's overfitting — increase dropout or reduce LSTM units.

---

## DAY 3 — Evaluate

### Step 8 — Core Metrics

```python
from sklearn.metrics import (
    classification_report,
    confusion_matrix,
    f1_score,
    accuracy_score
)
import numpy as np

# Predict on test set
y_pred_probs = model.predict(X_test)
y_pred = np.argmax(y_pred_probs, axis=1)
y_true = np.argmax(y_test, axis=1)

CLASS_NAMES = ['Critical', 'High', 'Low', 'Medium']  # adjust to your actual order

# Accuracy
accuracy = accuracy_score(y_true, y_pred)
print(f"Test Accuracy: {accuracy:.4f}")

# Macro F1 — your primary metric, report this prominently
macro_f1 = f1_score(y_true, y_pred, average='macro')
print(f"Macro F1-Score: {macro_f1:.4f}")

# Per-class breakdown
print("\nClassification Report:")
print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))
```

**Report both accuracy AND Macro F1** in your presentation. Macro F1 is more honest when classes are imbalanced — it treats each class equally regardless of size. Professors know this. Showing only accuracy is a red flag.

---

### Step 9 — Confusion Matrix

This is the most important chart in your presentation. It shows which classes the model confuses, which is a far more interesting story than a single accuracy number.

```python
import seaborn as sns
import matplotlib.pyplot as plt

cm = confusion_matrix(y_true, y_pred)

plt.figure(figsize=(8, 6))
sns.heatmap(
    cm,
    annot=True,
    fmt='d',
    cmap='Blues',
    xticklabels=CLASS_NAMES,
    yticklabels=CLASS_NAMES,
    linewidths=0.5
)
plt.title('Confusion Matrix — Test Set', fontsize=14, fontweight='bold', pad=15)
plt.ylabel('True Label', fontsize=12)
plt.xlabel('Predicted Label', fontsize=12)
plt.tight_layout()
plt.savefig(SAVE_DIR + 'confusion_matrix.png', dpi=150)
plt.show()
```

**What to say about the confusion matrix in your presentation:** A model confusing "High" with "Medium" is acceptable — they are adjacent priorities. A model confusing "Low" with "Critical" would be a serious flaw. If your model makes the first type of error, frame it as expected and reasonable. If it makes the second type, address it directly.

---

### Step 10 — Save Final Model

```python
# Save model
model.save(SAVE_DIR + 'ticket_classifier.keras')
print("Model saved.")

# Confirm tokenizer is still the original one from Person 2
# DO NOT refit it — just confirm it's the same file
import pickle
with open('models/tokenizer.pkl', 'rb') as f:
    tokenizer_check = pickle.load(f)
print(f"Tokenizer vocab size: {len(tokenizer_check.word_index)}")
```

---

## Handoff to Person 2 (Day 3 AM)

Send Person 2 a message with:

```
✅ Model ready. Files committed to /backend/models/:
   - ticket_classifier.keras
   - tokenizer.pkl (same file from Day 1, not refitted)

📊 Results for the presentation slides:
   - Test Accuracy:  XX.X%
   - Macro F1-Score: X.XXX
   - Best epoch: XX (stopped by EarlyStopping)
   - Class weights used: Yes/No

📁 Also in /backend/models/:
   - training_curves.png
   - confusion_matrix.png
```

---

## Troubleshooting

**Accuracy stuck below 50%** → Check that `y_train` is one-hot encoded correctly. Print `y_train[:3]` — each row should have exactly one `1.0`. If it's all integers, the encoding step was skipped.

**Loss goes to NaN after epoch 1** → Learning rate is too high. Change Adam to `learning_rate=1e-4`.

**Training very slow on Colab** → Confirm GPU is enabled: Runtime → Change runtime type → T4 GPU. Check with `tf.config.list_physical_devices('GPU')` — should return a non-empty list.

**Model always predicts the same class** → Class imbalance is severe. Add `class_weight=class_weight_dict` to `model.fit()` and make sure you computed it from the actual training labels.

**Val accuracy much lower than train accuracy (overfitting)** → Increase `Dropout` from 0.4 to 0.5, reduce `LSTM_UNITS` from 128 to 64, and let `EarlyStopping` run with `patience=3` instead of 5.

---

## Q&A — Likely Questions and Your Answers

**"Why Bidirectional LSTM and not a simpler model?"**
A standard LSTM only reads text forward. Bidirectional reads it both ways, capturing context that follows a word as well as what precedes it. For support tickets, the severity of an issue is often clarified at the end of the sentence — the Bidirectional architecture captures that.

**"How did you prevent overfitting?"**
Three mechanisms: Dropout layers (0.4 after LSTM, 0.3 after Dense), EarlyStopping with patience 5 on validation loss, and ReduceLROnPlateau which reduces the learning rate when progress stalls.

**"Why not use a pre-trained model like BERT?"**
BERT is a generative/transformer-based architecture outside the scope of this course, which requires predictive architectures (ANN, CNN, RNN/LSTM). Our Bi-LSTM with a trained embedding layer is the correct and well-justified choice for this task and dataset size.

**"What does the confusion matrix tell you?"**
It shows that most errors occur between adjacent priority levels (High/Medium or Medium/Low) — which is expected and acceptable. The model almost never confuses Critical with Low, which would be the most dangerous type of error in a real system.
