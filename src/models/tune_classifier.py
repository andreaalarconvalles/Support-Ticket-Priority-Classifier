import os
import numpy as np
import pickle
import tensorflow as tf
from sklearn.utils.class_weight import compute_class_weight
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.models import Model
from tensorflow.keras.layers import Input, Embedding, Bidirectional, LSTM, Dense, Dropout, Concatenate
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import keras_tuner as kt

# --- Configuration ---
DATA_DIR = 'backend/data/processed/'
SAVE_DIR = 'backend/models/'

VOCAB_SIZE = 5968
MAX_SEQUENCE_LENGTH = 57
NUM_CLASSES = 4
CLASS_NAMES = ['Low', 'Medium', 'High', 'Critical']

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

    # Combine train and val for Cross-Validation and Tuning
    X_cv_text = np.concatenate([X_train_text, X_val_text], axis=0)
    X_cv_tab = np.concatenate([X_train_tab, X_val_tab], axis=0)
    
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_train_int = np.argmax(y_train, axis=1)
        y_val_int = np.argmax(y_val, axis=1)
    else:
        y_train_int = y_train
        y_val_int = y_val
        
    y_cv = np.concatenate([y_train_int, y_val_int], axis=0)
    
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_test_int = np.argmax(y_test, axis=1)
    else:
        y_test_int = y_test

    print(f"Combined CV Set (X_cv_text) shape: {X_cv_text.shape}")
    print(f"Combined CV Set (X_cv_tab) shape: {X_cv_tab.shape}")
    print(f"Combined CV Set (y_cv) shape: {y_cv.shape}")
    print(f"Vocab size: {len(tokenizer.word_index)}")
    
    return X_cv_text, X_cv_tab, y_cv, X_test_text, X_test_tab, y_test_int, tokenizer

def get_class_weights(y):
    """Calculates class weights if there is an imbalance."""
    classes = np.unique(y)
    weights = compute_class_weight('balanced', classes=classes, y=y)
    class_weight_dict = dict(zip(classes, weights))
    return class_weight_dict

def build_model(hp, actual_vocab_size, tabular_dim):
    """Builds the Multi-Modal Bi-LSTM + Tabular model architecture with hyperparameter search space."""
    
    # Search Space Definitions
    hp_embedding_dim = hp.Choice('embedding_dim', values=[32, 64, 128])
    hp_lstm_units = hp.Choice('lstm_units', values=[64, 128, 256])
    hp_dropout_1 = hp.Float('dropout_1', min_value=0.2, max_value=0.6, step=0.1)
    
    hp_dense_tab = hp.Choice('dense_tab', values=[32, 64])
    hp_dropout_tab = hp.Float('dropout_tab', min_value=0.1, max_value=0.5, step=0.1)
    
    hp_dense_merged = hp.Choice('dense_merged', values=[32, 64, 128])
    hp_dropout_merged = hp.Float('dropout_merged', min_value=0.2, max_value=0.6, step=0.1)
    
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # 1. Text Pathway
    text_input = Input(shape=(MAX_SEQUENCE_LENGTH,), name='text_input')
    x_text = Embedding(
        input_dim=actual_vocab_size + 1,
        output_dim=hp_embedding_dim,
        input_length=MAX_SEQUENCE_LENGTH,
        mask_zero=True,
        trainable=True,
        name='embedding'
    )(text_input)
    x_text = Bidirectional(
        LSTM(hp_lstm_units, return_sequences=False, dropout=0.2, recurrent_dropout=0.1),
        name='bi_lstm'
    )(x_text)
    x_text = Dropout(hp_dropout_1, name='dropout_text')(x_text)

    # 2. Tabular Pathway
    tabular_input = Input(shape=(tabular_dim,), name='tabular_input')
    x_tab = Dense(hp_dense_tab, activation='relu', name='dense_tab_1')(tabular_input)
    x_tab = Dropout(hp_dropout_tab, name='dropout_tab_1')(x_tab)

    # 3. Concatenation and Final Layers
    merged = Concatenate(name='concat_layer')([x_text, x_tab])
    x_merged = Dense(hp_dense_merged, activation='relu', name='dense_merged_1')(merged)
    x_merged = Dropout(hp_dropout_merged, name='dropout_merged_1')(x_merged)

    outputs = Dense(NUM_CLASSES, activation='softmax', name='output')(x_merged)

    model = Model(inputs=[text_input, tabular_input], outputs=outputs)
    
    model.compile(
        optimizer=tf.keras.optimizers.Adam(learning_rate=hp_learning_rate),
        loss='sparse_categorical_crossentropy',
        metrics=['accuracy']
    )
    return model

def tune_hyperparameters(X_cv_text, X_cv_tab, y_cv, actual_vocab_size, tabular_dim):
    """Runs Keras Tuner to find the best hyperparameters."""
    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_model(hp, actual_vocab_size, tabular_dim),
        objective='val_accuracy',
        max_trials=10,  # Number of parameter combinations to try
        executions_per_trial=1,
        directory='backend/tuning_logs',
        project_name='ticket_classifier_multimodal',
        overwrite=False
    )
    
    tuner.search_space_summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]
    
    class_weights = get_class_weights(y_cv)
    
    print("\nStarting hyperparameter search...")
    tuner.search(
        [X_cv_text, X_cv_tab], y_cv,
        epochs=15,
        validation_split=0.2,  # Internal split just for tuning guidance
        class_weight=class_weights,
        callbacks=callbacks,
        batch_size=64,
        verbose=1
    )
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return tuner, best_hps

def run_cross_validation(best_hps, X_cv_text, X_cv_tab, y_cv, actual_vocab_size, tabular_dim, n_splits=5):
    """Evaluates the best architecture using Stratified K-Fold CV."""
    print(f"\n--- Running {n_splits}-Fold Stratified Cross Validation with best parameters ---")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold = 1
    cv_accuracies = []
    cv_f1s = []
    
    for train_idx, val_idx in skf.split(X_cv_text, y_cv):
        print(f"\nTraining Fold {fold}...")
        X_train_text_f, X_val_text_f = X_cv_text[train_idx], X_cv_text[val_idx]
        X_train_tab_f, X_val_tab_f = X_cv_tab[train_idx], X_cv_tab[val_idx]
        y_train_f, y_val_f = y_cv[train_idx], y_cv[val_idx]
        
        model = build_model(best_hps, actual_vocab_size, tabular_dim)
        class_weights = get_class_weights(y_train_f)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0)
        ]
        
        model.fit(
            [X_train_text_f, X_train_tab_f], y_train_f,
            validation_data=([X_val_text_f, X_val_tab_f], y_val_f),
            epochs=15,
            batch_size=64,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate fold
        y_pred_probs = model.predict([X_val_text_f, X_val_tab_f], verbose=0)
        y_pred = np.argmax(y_pred_probs, axis=1)
        
        acc = accuracy_score(y_val_f, y_pred)
        f1 = f1_score(y_val_f, y_pred, average='macro')
        
        cv_accuracies.append(acc)
        cv_f1s.append(f1)
        print(f"Fold {fold} - Accuracy: {acc:.4f}, Macro F1: {f1:.4f}")
        fold += 1
        
    print(f"\n--- Cross Validation Results ---")
    print(f"Average CV Accuracy:  {np.mean(cv_accuracies):.4f} (+/- {np.std(cv_accuracies):.4f})")
    print(f"Average CV Macro F1:  {np.mean(cv_f1s):.4f} (+/- {np.std(cv_f1s):.4f})")

def train_final_model_and_evaluate(best_hps, X_cv_text, X_cv_tab, y_cv, X_test_text, X_test_tab, y_test, actual_vocab_size, tabular_dim):
    """Trains on all CV data and tests on the holdout test set."""
    print("\n--- Training Final Model on Full CV Set ---")
    os.makedirs(SAVE_DIR, exist_ok=True)
    
    model = build_model(best_hps, actual_vocab_size, tabular_dim)
    class_weights = get_class_weights(y_cv)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    
    model.fit(
        [X_cv_text, X_cv_tab], y_cv,
        validation_split=0.1,
        epochs=30,
        batch_size=64,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n--- Evaluating Final Model on Holdout Test Set ---")
    y_pred_probs = model.predict([X_test_text, X_test_tab])
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    accuracy = accuracy_score(y_test, y_pred)
    macro_f1 = f1_score(y_test, y_pred, average='macro')
    
    print(f"\nFinal Test Accuracy:  {accuracy:.4f}")
    print(f"Final Test Macro F1: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_test, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_test, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, linewidths=0.5)
    plt.title('Confusion Matrix — Tuned Multi-Modal Model (Test Set)', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(os.path.join(SAVE_DIR, 'confusion_matrix_tuned_multimodal.png'), dpi=150)
    
    model.save(os.path.join(SAVE_DIR, 'ticket_classifier_tuned_multimodal.keras'))
    print("✅ Final Tuned Model saved as ticket_classifier_tuned_multimodal.keras")

if __name__ == "__main__":
    # 1. Load Data
    X_cv_text, X_cv_tab, y_cv, X_test_text, X_test_tab, y_test, tokenizer = load_data()
    actual_vocab_size = min(VOCAB_SIZE, len(tokenizer.word_index))
    tabular_dim = X_cv_tab.shape[1]
    
    # 2. Run Hyperparameter Tuning
    tuner, best_hps = tune_hyperparameters(X_cv_text, X_cv_tab, y_cv, actual_vocab_size, tabular_dim)
    
    # 3. Validate with Stratified K-Fold
    run_cross_validation(best_hps, X_cv_text, X_cv_tab, y_cv, actual_vocab_size, tabular_dim, n_splits=5)
    
    # 4. Train final model and evaluate
    train_final_model_and_evaluate(best_hps, X_cv_text, X_cv_tab, y_cv, X_test_text, X_test_tab, y_test, actual_vocab_size, tabular_dim)
    print("\n🎉 Tuning and Evaluation Complete!")
