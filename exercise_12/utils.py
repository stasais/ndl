"""
Utilities for Exercise 12 - LSTM IMDB Sentiment Classification
"""
import keras
import numpy as np
from typing import Tuple


# =============================================================================
# Data Loading
# =============================================================================

def load_imdb_data(
    num_words: int = 10000,
    maxlen: int = 500,
    test_split: float = 0.5
) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, np.ndarray, dict]:
    """
    Load IMDB dataset from keras.datasets.
    
    Args:
        num_words: Maximum vocabulary size
        maxlen: Maximum sequence length (sequences will be padded/truncated)
        test_split: Fraction of test data to use as hold-out set
        
    Returns:
        X_train, y_train: Training data
        X_val, y_val: Validation data (split from test)  
        X_test, y_test: Hold-out test data
        word_index: Word to index mapping
    """

    (X_train, y_train), (X_test_full, y_test_full) = keras.datasets.imdb.load_data(
        num_words=num_words
    )
    
    X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen, padding='post')
    X_test_full = keras.preprocessing.sequence.pad_sequences(X_test_full, maxlen=maxlen, padding='post')
    

    split_idx = int(len(X_test_full) * test_split)
    X_val, X_test = X_test_full[:split_idx], X_test_full[split_idx:]
    y_val, y_test = y_test_full[:split_idx], y_test_full[split_idx:]
    

    word_index = keras.datasets.imdb.get_word_index()
    
    print(f"Training samples:   {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Test samples:       {len(X_test):,}")
    print(f"Vocabulary size:    {num_words:,}")
    print(f"Sequence length:    {maxlen}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, word_index


# =============================================================================
# GloVe Embeddings
# =============================================================================

def download_glove_embeddings(
    glove_dim: int = 100,
    cache_dir: str = None
) -> str:
    """
    Download GloVe embeddings.
    
    Args:
        glove_dim: Embedding dimension (50, 100, 200, or 300)
        cache_dir: Directory to cache the embeddings
        
    Returns:
        Path to the downloaded embeddings file
    """
    import os
    import urllib.request
    import zipfile
    
    if cache_dir is None:
        cache_dir = os.path.expanduser('~/.keras/datasets')
    
    os.makedirs(cache_dir, exist_ok=True)
    
    glove_file = os.path.join(cache_dir, f'glove.6B.{glove_dim}d.txt')
    
    if os.path.exists(glove_file):
        print(f"GloVe embeddings already exist at {glove_file}")
        return glove_file
    
    zip_path = os.path.join(cache_dir, 'glove.6B.zip')
    
    if not os.path.exists(zip_path):
        print("Downloading GloVe embeddings (822 MB)...")
        url = "https://nlp.stanford.edu/data/glove.6B.zip"
        urllib.request.urlretrieve(url, zip_path)
        print("Download complete!")
    
    print("Extracting embeddings...")
    with zipfile.ZipFile(zip_path, 'r') as zip_ref:
        zip_ref.extract(f'glove.6B.{glove_dim}d.txt', cache_dir)
    
    print(f"GloVe embeddings saved to {glove_file}")
    return glove_file


def create_embedding_matrix_from_glove(
    glove_path: str,
    word_index: dict,
    embedding_dim: int = 100,
    num_words: int = 10000
) -> np.ndarray:
    """
    Create embedding matrix from GloVe embeddings.
    
    Args:
        glove_path: Path to GloVe embeddings file
        word_index: Dictionary mapping words to indices (from keras IMDB dataset)
        embedding_dim: Dimension of embeddings
        num_words: Maximum vocabulary size
        
    Returns:
        Embedding matrix of shape (num_words, embedding_dim)
    """
    print(f"Loading GloVe embeddings from {glove_path}...")
    
    # Load GloVe embeddings
    embeddings_index = {}
    with open(glove_path, 'r', encoding='utf-8') as f:
        for line in f:
            values = line.split()
            word = values[0]
            try:
                coefs = np.asarray(values[1:], dtype='float32')
                if len(coefs) == embedding_dim:
                    embeddings_index[word] = coefs
            except ValueError:
                continue
    
    print(f"Loaded {len(embeddings_index):,} word vectors")
    
    # Create embedding matrix
    # Index 0 is reserved for padding, 1 for start, 2 for OOV
    embedding_matrix = np.zeros((num_words, embedding_dim))
    found_words = 0
    missing_words = []
    
    for word, i in word_index.items():
        if i >= num_words - 3:  # Account for special tokens
            continue
        embedding_vector = embeddings_index.get(word)
        if embedding_vector is not None:
            # Shift by 3 because of special tokens (PAD, START, OOV)
            embedding_matrix[i + 3] = embedding_vector
            found_words += 1
        else:
            missing_words.append(word)
    
    coverage = 100 * found_words / min(len(word_index), num_words - 3)
    print(f"Matched {found_words:,} words ({coverage:.1f}% coverage)")
    
    if len(missing_words) <= 10:
        print(f"Missing words: {missing_words}")
    else:
        print(f"Sample missing words: {missing_words[:10]}")
    
    return embedding_matrix


# =============================================================================
# Model Training
# =============================================================================

def train_model(
    model: keras.Model,
    X_train: np.ndarray,
    y_train: np.ndarray,
    X_val: np.ndarray,
    y_val: np.ndarray,
    model_name: str,
    epochs: int = 10,
    batch_size: int = 64,
    early_stopping: bool = True,
    patience: int = 3,
    verbose: int = 1
) -> Tuple[keras.Model, keras.callbacks.History]:
    """
    Train a model with TensorBoard logging and optional early stopping.
    
    Args:
        model: Keras model to train
        X_train, y_train: Training data
        X_val, y_val: Validation data
        model_name: Name for logging
        epochs: Number of training epochs
        batch_size: Batch size
        early_stopping: Whether to use early stopping
        patience: Early stopping patience
        verbose: Verbosity level
        
    Returns:
        Trained model and training history
    """
    log_dir = f"logs/{model_name}"
    tensorboard_callback = keras.callbacks.TensorBoard(
        log_dir=log_dir, 
        histogram_freq=1
    )
    
    callbacks = [tensorboard_callback]
    
    if early_stopping:
        early_stop = keras.callbacks.EarlyStopping(
            monitor='val_loss',
            patience=patience,
            restore_best_weights=True,
            verbose=1
        )
        callbacks.append(early_stop)
    
    history = model.fit(
        X_train, y_train,
        validation_data=(X_val, y_val),
        epochs=epochs,
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    
    return model, history


# =============================================================================
# Model Evaluation
# =============================================================================

def eval_binary_classification(
    model: keras.Model,
    X_test: np.ndarray,
    y_test: np.ndarray,
    model_name: str = None,
    threshold: float = 0.5
) -> Tuple[float, float]:
    """
    Evaluate binary classification model and display metrics.
    
    Args:
        model: Trained Keras model
        X_test: Test features
        y_test: Test labels
        model_name: Name for display
        threshold: Classification threshold
        
    Returns:
        test_loss, test_accuracy
    """
    import matplotlib.pyplot as plt
    from sklearn.metrics import (
        confusion_matrix, classification_report, 
        ConfusionMatrixDisplay, roc_curve, auc
    )
    
    # Get predictions
    y_pred_proba = model.predict(X_test, verbose=0).flatten()
    y_pred = (y_pred_proba >= threshold).astype(int)
    
    # Evaluate
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    
    print(f"\n{'='*60}")
    print(f"  {model_name or 'Model'} - Binary Classification Results")
    print(f"{'='*60}")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{'='*60}\n")
    
    # Classification report
    class_names = ['Negative', 'Positive']
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    # Confusion matrix
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cm = confusion_matrix(y_test, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title(f"{model_name or 'Model'} - Confusion Matrix")
    
    # ROC curve
    fpr, tpr, _ = roc_curve(y_test, y_pred_proba)
    roc_auc = auc(fpr, tpr)
    
    axes[1].plot(fpr, tpr, color='darkorange', lw=2, 
                 label=f'ROC curve (AUC = {roc_auc:.3f})')
    axes[1].plot([0, 1], [0, 1], color='navy', lw=2, linestyle='--')
    axes[1].set_xlim([0.0, 1.0])
    axes[1].set_ylim([0.0, 1.05])
    axes[1].set_xlabel('False Positive Rate')
    axes[1].set_ylabel('True Positive Rate')
    axes[1].set_title(f"{model_name or 'Model'} - ROC Curve")
    axes[1].legend(loc='lower right')
    
    plt.tight_layout()
    plt.show()
    
    return test_loss, test_acc


def compare_models(
    results: dict,
    metric: str = 'accuracy'
) -> None:
    """
    Compare multiple models visually.
    
    Args:
        results: Dictionary with model names as keys and 
                 (loss, accuracy) tuples as values
        metric: Which metric to highlight ('accuracy' or 'loss')
    """
    import matplotlib.pyplot as plt
    
    names = list(results.keys())
    losses = [r[0] for r in results.values()]
    accuracies = [r[1] for r in results.values()]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    
    # Loss comparison
    bars1 = axes[0].bar(names, losses, color=colors)
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Model Comparison - Test Loss')
    axes[0].set_ylim(0, max(losses) * 1.2)
    for bar, loss in zip(bars1, losses):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{loss:.4f}', ha='center', va='bottom', fontsize=10)
    
    # Accuracy comparison
    bars2 = axes[1].bar(names, [a * 100 for a in accuracies], color=colors)
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Model Comparison - Test Accuracy')
    axes[1].set_ylim(0, 100)
    for bar, acc in zip(bars2, accuracies):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{acc*100:.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()


def plot_training_history(history: keras.callbacks.History, title: str = None) -> None:
    """Plot training history with loss and accuracy curves."""
    import matplotlib.pyplot as plt
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    # Loss
    axes[0].plot(history.history['loss'], label='Training Loss', linewidth=2)
    axes[0].plot(history.history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{title + " - " if title else ""}Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    # Accuracy
    axes[1].plot(history.history['accuracy'], label='Training Accuracy', linewidth=2)
    axes[1].plot(history.history['val_accuracy'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{title + " - " if title else ""}Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()
