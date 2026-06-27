import os
import numpy as np
import pickle
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ModelCheckpoint, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score

# --- Configuration ---
DATA_DIR = 'backend/data/processed/'
SAVE_DIR = 'backend/models/'

VOCAB_SIZE = 5968
MAX_SEQUENCE_LENGTH = 57
NUM_CLASSES = 4
CLASS_NAMES = ['Low', 'Medium', 'High', 'Critical']

# Hyperparameters
EMBEDDING_DIM = 64
LSTM_UNITS = 128

def load_data():
    """Loads preprocessed data and tokenizer."""
    print("Loading data from", DATA_DIR)
    X_train_text = np.load(os.path.join(DATA_DIR, 'X_train_text.npy'))
    X_val_text   = np.load(os.path.join(DATA_DIR, 'X_val_text.npy'))
    X_test_text  = np.load(os.path.join(DATA_DIR, 'X_test_text.npy'))
    
    X_train_tab = np.load(os.path.join(DATA_DIR, 'X_train_tabular.npy'))
    X_val_tab   = np.load(os.path.join(DATA_DIR, 'X_val_tabular.npy'))
    X_test_tab  = np.load(os.path.join(DATA_DIR, 'X_test_tabular.npy'))
    
    y_train = np.load(os.path.join(DATA_DIR, 'y_train.npy'))
    y_val   = np.load(os.path.join(DATA_DIR, 'y_val.npy'))
    y_test  = np.load(os.path.join(DATA_DIR, 'y_test.npy'))

    with open(os.path.join(DATA_DIR, 'tokenizer.pkl'), 'rb') as f:
        tokenizer = pickle.load(f)

    print(f"Loaded X_train_text shape: {X_train_text.shape}")
    print(f"Loaded X_train_tab shape: {X_train_tab.shape}")
    print(f"Loaded y_train shape: {y_train.shape}")
    print(f"Vocab size: {len(tokenizer.word_index)}")
    
    return X_train_text, X_train_tab, X_val_text, X_val_tab, X_test_text, X_test_tab, y_train, y_val, y_test, tokenizer

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

def build_model(actual_vocab_size, tabular_dim):
    """Builds the Multi-Modal Bi-LSTM + Tabular Dense model architecture."""
    # 1. Text Input Pathway
    text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='text_input')
    x_text = Embedding(
        input_dim=actual_vocab_size + 1,
        output_dim=EMBEDDING_DIM,
        input_length=MAX_SEQUENCE_LENGTH,
        mask_zero=True,  # Crucial for ignoring the zero padding!
        trainable=True,
        name='embedding'
    )(text_input)
    x_text = Bidirectional(
        LSTM(LSTM_UNITS, return_sequences=False, dropout=0.2, recurrent_dropout=0.1),
        name='bi_lstm'
    )(x_text)
    x_text = Dropout(0.4, name='dropout_text')(x_text)

    # 2. Tabular Input Pathway
    tabular_input = Input(shape=(tabular_dim,), name='tabular_input')
    x_tab = Dense(64, activation='relu', name='dense_tab_1')(tabular_input)
    x_tab = Dropout(0.3, name='dropout_tab_1')(x_tab)

    # 3. Concatenation and Final Layers
    merged = Concatenate(name='concat_layer')([x_text, x_tab])
    x_merged = Dense(64, activation='relu', name='dense_merged_1')(merged)
    x_merged = Dropout(0.3, name='dropout_merged_1')(x_merged)

    outputs = Dense(NUM_CLASSES, activation='softmax', name='output')(x_merged)

    model = Model(inputs=[text_input, tabular_input], outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=1e-3),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def train_model(model, X_train_text, X_train_tab, y_train, X_val_text, X_val_tab, y_val, class_weight_dict):
    """Trains the multi-modal model with callbacks."""
    os.makedirs(SAVE_DIR, exist_ok=True)
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ModelCheckpoint(filepath=os.path.join(SAVE_DIR, 'ticket_classifier.keras'), monitor='val_loss', save_best_only=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]

    history = model.fit(
        [X_train_text, X_train_tab], y_train,
        validation_data=([X_val_text, X_val_tab], y_val),
        epochs=30,
        batch_size=64,
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
    plt.savefig(os.path.join(SAVE_DIR, 'training_curves.png'), dpi=150)
    print("Saved training curves plot.")

def evaluate_and_save(model, X_test_text, X_test_tab, y_test):
    """Evaluates the model, prints metrics, saves confusion matrix and model."""
    y_pred_probs = model.predict([X_test_text, X_test_tab])
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
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix.png'), dpi=150)
    print("Saved confusion matrix plot.")

    model.save(os.path.join(SAVE_DIR, 'ticket_classifier.keras'))
    print("✅ Final Model saved as ticket_classifier.keras")

if __name__ == "__main__":
    # 1. Load the real data
    X_train_text, X_train_tab, X_val_text, X_val_tab, X_test_text, X_test_tab, y_train, y_val, y_test, tokenizer = load_data()
    
    # 2. Handle class weights
    class_weights = get_class_weights(y_train)
    
    # 3. Build Model
    actual_vocab_size = min(VOCAB_SIZE, len(tokenizer.word_index))
    tabular_dim = X_train_tab.shape[1]
    model = build_model(actual_vocab_size, tabular_dim)
    model.summary()
    
    # 4. Train Model
    history = train_model(model, X_train_text, X_train_tab, y_train, X_val_text, X_val_tab, y_val, class_weights)
    
    # 5. Evaluate and Plot
    plot_history(history)
    evaluate_and_save(model, X_test_text, X_test_tab, y_test)
    print("\n🎉 ML Pipeline Complete!")
