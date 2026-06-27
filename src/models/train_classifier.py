import os
import random
import numpy as np
import pickle
import sys
from pathlib import Path
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# Fix import path for config
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import PROCESSED_DATA_DIR, MODELS_DIR

# --- Configuration ---
VOCAB_SIZE = 5726
MAX_SEQUENCE_LENGTH = 57
NUM_CLASSES = 3
CLASS_NAMES = ['Low', 'Medium', 'High']

def set_global_seed(seed=42):
    """Sets random seeds for reproducibility."""
    os.environ['PYTHONHASHSEED'] = str(seed)
    random.seed(seed)
    np.random.seed(seed)
    tf.random.set_seed(seed)

def load_data():
    """Loads preprocessed data and vocabulary."""
    print("Loading data from", PROCESSED_DATA_DIR)
    X_train_text = np.load(PROCESSED_DATA_DIR / 'X_train_text.npy')
    X_val_text   = np.load(PROCESSED_DATA_DIR / 'X_val_text.npy')
    X_test_text  = np.load(PROCESSED_DATA_DIR / 'X_test_text.npy')
    
    y_train = np.load(PROCESSED_DATA_DIR / 'y_train.npy')
    y_val   = np.load(PROCESSED_DATA_DIR / 'y_val.npy')
    y_test  = np.load(PROCESSED_DATA_DIR / 'y_test.npy')

    with open(PROCESSED_DATA_DIR / 'vocab.pkl', 'rb') as f:
        vocab = pickle.load(f)

    print(f"Loaded X_train_text shape: {X_train_text.shape}")
    print(f"Loaded y_train shape: {y_train.shape}")
    print(f"Vocab size: {len(vocab)}")
    
    return X_train_text, X_val_text, X_test_text, y_train, y_val, y_test, vocab

def create_dataset(X, y, batch_size=64, is_training=True):
    """Converts numpy arrays to an optimized tf.data.Dataset."""
    dataset = tf.data.Dataset.from_tensor_slices((X, y))
    if is_training:
        dataset = dataset.shuffle(buffer_size=len(X), seed=42)
    dataset = dataset.batch(batch_size).prefetch(tf.data.AUTOTUNE)
    return dataset

def get_class_weights(y_train):
    """Calculates class weights if there is an imbalance."""
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_int = np.argmax(y_train, axis=1)
    else:
        y_int = y_train
    classes = np.unique(y_int)
    weights = compute_class_weight('balanced', classes=classes, y=y_int)
    class_weight_dict = dict(zip(classes, weights))
    print("Computed Class Weights:", class_weight_dict)
    return class_weight_dict

def build_model(actual_vocab_size, 
                embedding_dim=64, 
                lstm_units=128, 
                dropout_rate=0.4, 
                learning_rate=1e-3):
    """Builds the Text Bi-LSTM model architecture (Single Source of Truth)."""
    text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='text_input')
    
    x = Embedding(
        input_dim=actual_vocab_size + 1,
        output_dim=embedding_dim,
        input_length=MAX_SEQUENCE_LENGTH,
        mask_zero=True,  # Crucial for ignoring the zero padding!
        trainable=True,
        name='embedding'
    )(text_input)
    
    x = Bidirectional(
        LSTM(lstm_units, return_sequences=False, dropout=0.2, recurrent_dropout=0.1),
        name='bi_lstm'
    )(x)
    
    x = Dropout(dropout_rate, name='dropout_text')(x)
    
    outputs = Dense(NUM_CLASSES, activation='softmax', name='output')(x)

    model = Model(inputs=text_input, outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model, train_dataset, val_dataset, class_weight_dict):
    """Trains the model with callbacks."""
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=str(MODELS_DIR / 'ticket_classifier.keras'), monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    history = model.fit(
        train_dataset,
        validation_data=val_dataset,
        epochs=30,
        callbacks=callbacks,
        class_weight=class_weight_dict,
        verbose=1
    )
    return history

def plot_history(history):
    """Plots and saves the training history curves."""
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
    plt.savefig(MODELS_DIR / 'training_curves.png', dpi=150)
    print("Saved training curves plot.")

def evaluate_and_save(model, test_dataset, y_test):
    """Evaluates the model, prints metrics, saves confusion matrix and model."""
    y_pred_probs = model.predict(test_dataset)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"\n--- EVALUATION RESULTS ---")
    print(f"Test Accuracy:  {accuracy:.4f}")
    print(f"Macro F1-Score: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, linewidths=0.5)
    plt.title('Confusion Matrix — Test Set', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'confusion_matrix.png', dpi=150)
    print("Saved confusion matrix plot.")

    model.save(MODELS_DIR / 'ticket_classifier.keras')
    print("✅ Final Model saved as ticket_classifier.keras")

if __name__ == "__main__":
    # 0. Set seed for reproducibility
    set_global_seed(42)

    # 1. Load the real data
    X_train_text, X_val_text, X_test_text, y_train, y_val, y_test, vocab = load_data()
    
    # 2. Convert to tf.data.Dataset
    BATCH_SIZE = 64
    train_dataset = create_dataset(X_train_text, y_train, batch_size=BATCH_SIZE, is_training=True)
    val_dataset = create_dataset(X_val_text, y_val, batch_size=BATCH_SIZE, is_training=False)
    test_dataset = create_dataset(X_test_text, y_test, batch_size=BATCH_SIZE, is_training=False)
    
    # 3. Handle class weights
    class_weights = get_class_weights(y_train)
    
    # 4. Build Model
    actual_vocab_size = min(VOCAB_SIZE, len(vocab))
    model = build_model(actual_vocab_size)
    model.summary()
    
    # 5. Train Model
    history = train_model(model, train_dataset, val_dataset, class_weights)
    
    # 6. Evaluate and Plot
    plot_history(history)
    # y_test is passed to compare against predictions
    evaluate_and_save(model, test_dataset, y_test)
    print("\n🎉 ML Pipeline Complete!")
