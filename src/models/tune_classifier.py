import os
import numpy as np
import tensorflow as tf
from sklearn.model_selection import StratifiedKFold
from tensorflow.keras.callbacks import EarlyStopping, ReduceLROnPlateau
import matplotlib.pyplot as plt
import seaborn as sns
from sklearn.metrics import classification_report, confusion_matrix, f1_score, accuracy_score
import keras_tuner as kt
import sys
from pathlib import Path

# Fix import path for config
sys.path.append(str(Path(__file__).resolve().parent.parent.parent))
from src.config import MODELS_DIR

# Import single source of truth from train_classifier
from src.models.train_classifier import (
    set_global_seed,
    load_data, 
    create_dataset, 
    get_class_weights, 
    build_model,
    VOCAB_SIZE,
    CLASS_NAMES
)

def combine_for_cv(X_train_text, X_val_text, y_train, y_val):
    """Combines train and val sets for Cross-Validation and Tuning."""
    X_cv_text = np.concatenate([X_train_text, X_val_text], axis=0)
    
    if len(y_train.shape) > 1 and y_train.shape[1] > 1:
        y_train_int = np.argmax(y_train, axis=1)
        y_val_int = np.argmax(y_val, axis=1)
    else:
        y_train_int = y_train
        y_val_int = y_val
        
    y_cv = np.concatenate([y_train_int, y_val_int], axis=0)
    
    print(f"Combined CV Set (X_cv_text) shape: {X_cv_text.shape}")
    print(f"Combined CV Set (y_cv) shape: {y_cv.shape}")
    return X_cv_text, y_cv

def build_model_tuner(hp, actual_vocab_size):
    """Builds the model architecture using hyperparameter search space."""
    
    # Search Space Definitions
    hp_embedding_dim = hp.Choice('embedding_dim', values=[32, 64, 128])
    hp_lstm_units = hp.Choice('lstm_units', values=[64, 128, 256])
    hp_dropout = hp.Float('dropout_1', min_value=0.2, max_value=0.6, step=0.1)
    hp_learning_rate = hp.Choice('learning_rate', values=[1e-2, 1e-3, 1e-4])

    # Defer to the single source of truth
    return build_model(
        actual_vocab_size=actual_vocab_size,
        embedding_dim=hp_embedding_dim,
        lstm_units=hp_lstm_units,
        dropout_rate=hp_dropout,
        learning_rate=hp_learning_rate
    )

def tune_hyperparameters(X_cv_text, y_cv, actual_vocab_size):
    """Runs Keras Tuner to find the best hyperparameters."""
    tuner = kt.RandomSearch(
        hypermodel=lambda hp: build_model_tuner(hp, actual_vocab_size),
        objective='val_accuracy',
        max_trials=10,  # Number of parameter combinations to try
        executions_per_trial=1,
        directory='tuning_logs',
        project_name='ticket_classifier_text_only',
        overwrite=False,
        seed=42
    )
    
    tuner.search_space_summary()
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True)
    ]
    
    class_weights = get_class_weights(y_cv)
    
    print("\nStarting hyperparameter search...")
    # Using numpy arrays here because Keras Tuner's validation_split splits automatically
    tuner.search(
        X_cv_text, y_cv,
        epochs=15,
        validation_split=0.2,  # Internal split just for tuning guidance
        class_weight=class_weights,
        callbacks=callbacks,
        batch_size=64,
        verbose=1
    )
    
    best_hps = tuner.get_best_hyperparameters(num_trials=1)[0]
    return tuner, best_hps

def run_cross_validation(best_hps, X_cv_text, y_cv, actual_vocab_size, n_splits=5):
    """Evaluates the best architecture using Stratified K-Fold CV and tf.data.Dataset."""
    print(f"\n--- Running {n_splits}-Fold Stratified Cross Validation with best parameters ---")
    skf = StratifiedKFold(n_splits=n_splits, shuffle=True, random_state=42)
    
    fold = 1
    cv_accuracies = []
    cv_f1s = []
    
    for train_idx, val_idx in skf.split(X_cv_text, y_cv):
        print(f"\nTraining Fold {fold}...")
        X_train_text_f, X_val_text_f = X_cv_text[train_idx], X_cv_text[val_idx]
        y_train_f, y_val_f = y_cv[train_idx], y_cv[val_idx]
        
        # Convert to tf.data.Dataset for training
        train_ds = create_dataset(X_train_text_f, y_train_f, batch_size=64, is_training=True)
        val_ds = create_dataset(X_val_text_f, y_val_f, batch_size=64, is_training=False)
        
        # Reconstruct best model using our single source of truth
        model = build_model(
            actual_vocab_size=actual_vocab_size,
            embedding_dim=best_hps.get('embedding_dim'),
            lstm_units=best_hps.get('lstm_units'),
            dropout_rate=best_hps.get('dropout_1'),
            learning_rate=best_hps.get('learning_rate')
        )
        
        class_weights = get_class_weights(y_train_f)
        
        callbacks = [
            EarlyStopping(monitor='val_loss', patience=3, restore_best_weights=True),
            ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=2, verbose=0)
        ]
        
        model.fit(
            train_ds,
            validation_data=val_ds,
            epochs=15,
            class_weight=class_weights,
            callbacks=callbacks,
            verbose=0
        )
        
        # Evaluate fold
        y_pred_probs = model.predict(val_ds, verbose=0)
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

def train_final_model_and_evaluate(best_hps, X_train_text, y_train, X_val_text, y_val, X_test_text, y_test, actual_vocab_size):
    """Trains the final model monitoring validation loss for early stopping, then evaluates on test set."""
    print("\n--- Training Final Model ---")
    MODELS_DIR.mkdir(parents=True, exist_ok=True)
    
    train_ds = create_dataset(X_train_text, y_train, batch_size=64, is_training=True)
    val_ds = create_dataset(X_val_text, y_val, batch_size=64, is_training=False)
    test_ds = create_dataset(X_test_text, y_test, batch_size=64, is_training=False)
    
    model = build_model(
        actual_vocab_size=actual_vocab_size,
        embedding_dim=best_hps.get('embedding_dim'),
        lstm_units=best_hps.get('lstm_units'),
        dropout_rate=best_hps.get('dropout_1'),
        learning_rate=best_hps.get('learning_rate')
    )
    
    class_weights = get_class_weights(y_train)
    
    callbacks = [
        EarlyStopping(monitor='val_loss', patience=5, restore_best_weights=True, verbose=1),
        ReduceLROnPlateau(monitor='val_loss', factor=0.5, patience=3, verbose=1)
    ]
    
    model.fit(
        train_ds,
        validation_data=val_ds,
        epochs=30,
        class_weight=class_weights,
        callbacks=callbacks,
        verbose=1
    )
    
    print("\n--- Evaluating Final Model on Holdout Test Set ---")
    y_pred_probs = model.predict(test_ds)
    y_pred = np.argmax(y_pred_probs, axis=1)
    
    if len(y_test.shape) > 1 and y_test.shape[1] > 1:
        y_true = np.argmax(y_test, axis=1)
    else:
        y_true = y_test

    accuracy = accuracy_score(y_true, y_pred)
    macro_f1 = f1_score(y_true, y_pred, average='macro')
    
    print(f"\nFinal Test Accuracy:  {accuracy:.4f}")
    print(f"Final Test Macro F1: {macro_f1:.4f}")
    print("\nClassification Report:")
    print(classification_report(y_true, y_pred, target_names=CLASS_NAMES))

    cm = confusion_matrix(y_true, y_pred)
    plt.figure(figsize=(8, 6))
    sns.heatmap(cm, annot=True, fmt='d', cmap='Blues', xticklabels=CLASS_NAMES, yticklabels=CLASS_NAMES, linewidths=0.5)
    plt.title('Confusion Matrix — Tuned Model (Test Set)', fontsize=14, fontweight='bold', pad=15)
    plt.ylabel('True Label', fontsize=12)
    plt.xlabel('Predicted Label', fontsize=12)
    plt.tight_layout()
    plt.savefig(MODELS_DIR / 'confusion_matrix_tuned.png', dpi=150)
    
    model.save(MODELS_DIR / 'ticket_classifier_tuned.keras')
    print("✅ Final Tuned Model saved as ticket_classifier_tuned.keras")

if __name__ == "__main__":
    # 0. Set seed for reproducibility
    set_global_seed(42)

    # 1. Load Data (Using single source of truth)
    X_train_text, X_val_text, X_test_text, y_train, y_val, y_test, vocab = load_data()
    actual_vocab_size = min(VOCAB_SIZE, len(vocab))
    
    # Combine train and val for tuning
    X_cv_text, y_cv = combine_for_cv(X_train_text, X_val_text, y_train, y_val)
    
    # 2. Run Hyperparameter Tuning
    tuner, best_hps = tune_hyperparameters(X_cv_text, y_cv, actual_vocab_size)
    print(f"Optimal config: Embedding {best_hps.get('embedding_dim')}, LSTM {best_hps.get('lstm_units')}, LR {best_hps.get('learning_rate')}")

    # 3. Validate with Stratified K-Fold
    run_cross_validation(best_hps, X_cv_text, y_cv, actual_vocab_size, n_splits=5)
    
    # 4. Train final model and evaluate
    train_final_model_and_evaluate(best_hps, X_train_text, y_train, X_val_text, y_val, X_test_text, y_test, actual_vocab_size)
    print("\n🎉 Tuning and Evaluation Complete!")
