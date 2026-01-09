"""
PyTorch utilities for Exercise 12 - xLSTM IMDB Sentiment Classification
"""
import torch
import torch.nn as nn
import numpy as np
from typing import Tuple
import matplotlib.pyplot as plt


def load_imdb_data_torch(
    num_words: int = 10000,
    maxlen: int = 500,
    test_split: float = 0.5
) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, torch.Tensor, dict]:
    """Load IMDB dataset and convert to PyTorch tensors."""
    import keras
    
    (X_train, y_train), (X_test_full, y_test_full) = keras.datasets.imdb.load_data(
        num_words=num_words
    )
    
    X_train = keras.preprocessing.sequence.pad_sequences(X_train, maxlen=maxlen, padding='post')
    X_test_full = keras.preprocessing.sequence.pad_sequences(X_test_full, maxlen=maxlen, padding='post')
    
    split_idx = int(len(X_test_full) * test_split)
    X_val, X_test = X_test_full[:split_idx], X_test_full[split_idx:]
    y_val, y_test = y_test_full[:split_idx], y_test_full[split_idx:]
    
    X_train = torch.tensor(X_train, dtype=torch.long)
    y_train = torch.tensor(y_train, dtype=torch.float32)
    X_val = torch.tensor(X_val, dtype=torch.long)
    y_val = torch.tensor(y_val, dtype=torch.float32)
    X_test = torch.tensor(X_test, dtype=torch.long)
    y_test = torch.tensor(y_test, dtype=torch.float32)
    
    word_index = keras.datasets.imdb.get_word_index()
    
    print(f"Training samples:   {len(X_train):,}")
    print(f"Validation samples: {len(X_val):,}")
    print(f"Test samples:       {len(X_test):,}")
    print(f"Vocabulary size:    {num_words:,}")
    print(f"Sequence length:    {maxlen}")
    
    return X_train, y_train, X_val, y_val, X_test, y_test, word_index


def create_dataloaders(
    X_train: torch.Tensor,
    y_train: torch.Tensor,
    X_val: torch.Tensor,
    y_val: torch.Tensor,
    batch_size: int = 64
) -> Tuple[torch.utils.data.DataLoader, torch.utils.data.DataLoader]:
    """Create PyTorch DataLoaders for training and validation."""
    train_dataset = torch.utils.data.TensorDataset(X_train, y_train)
    val_dataset = torch.utils.data.TensorDataset(X_val, y_val)
    
    train_loader = torch.utils.data.DataLoader(
        train_dataset, batch_size=batch_size, shuffle=True
    )
    val_loader = torch.utils.data.DataLoader(
        val_dataset, batch_size=batch_size, shuffle=False
    )
    
    return train_loader, val_loader


class LSTMClassifier(nn.Module):
    """Standard LSTM classifier."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, hidden_dim: int, padding_idx: int = 0):
        super().__init__()
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        self.lstm = nn.LSTM(embedding_dim, hidden_dim, batch_first=True)
        self.fc = nn.Linear(hidden_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        _, (hidden, _) = self.lstm(embedded)
        out = self.fc(hidden.squeeze(0))
        return self.sigmoid(out)


class mLSTMClassifier(nn.Module):
    """mLSTM-based classifier using xLSTM library."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, num_heads: int = 4, padding_idx: int = 0):
        super().__init__()
        from xlstm import (
            xLSTMBlockStack,
            xLSTMBlockStackConfig,
            mLSTMBlockConfig,
            mLSTMLayerConfig,
        )
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        cfg = xLSTMBlockStackConfig(
            mlstm_block=mLSTMBlockConfig(
                mlstm=mLSTMLayerConfig(
                    conv1d_kernel_size=4,
                    qkv_proj_blocksize=4,
                    num_heads=num_heads
                )
            ),
            context_length=512,
            num_blocks=1,
            embedding_dim=embedding_dim,
            slstm_at=[],
        )
        
        self.xlstm = xLSTMBlockStack(cfg)
        self.fc = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        xlstm_out = self.xlstm(embedded)
        last_hidden = xlstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return self.sigmoid(out)


class sLSTMClassifier(nn.Module):
    """sLSTM-based classifier using xLSTM library."""
    
    def __init__(self, vocab_size: int, embedding_dim: int, num_heads: int = 4, padding_idx: int = 0, backend: str = "vanilla"):
        super().__init__()
        from xlstm import (
            xLSTMBlockStack,
            xLSTMBlockStackConfig,
            sLSTMBlockConfig,
            sLSTMLayerConfig,
            FeedForwardConfig,
        )
        
        self.embedding = nn.Embedding(vocab_size, embedding_dim, padding_idx=padding_idx)
        
        cfg = xLSTMBlockStackConfig(
            slstm_block=sLSTMBlockConfig(
                slstm=sLSTMLayerConfig(
                    backend=backend,
                    num_heads=num_heads,
                    conv1d_kernel_size=4,
                    bias_init="powerlaw_blockdependent",
                ),
                feedforward=FeedForwardConfig(proj_factor=1.3, act_fn="gelu"),
            ),
            context_length=512,
            num_blocks=1,
            embedding_dim=embedding_dim,
            slstm_at=[0],
        )
        
        self.xlstm = xLSTMBlockStack(cfg)
        self.fc = nn.Linear(embedding_dim, 1)
        self.sigmoid = nn.Sigmoid()
        
    def forward(self, x):
        embedded = self.embedding(x)
        xlstm_out = self.xlstm(embedded)
        last_hidden = xlstm_out[:, -1, :]
        out = self.fc(last_hidden)
        return self.sigmoid(out)


def train_epoch(model, train_loader, optimizer, criterion, device):
    """Train for one epoch."""
    model.train()
    total_loss = 0
    correct = 0
    total = 0
    
    for X_batch, y_batch in train_loader:
        X_batch, y_batch = X_batch.to(device), y_batch.to(device)
        
        optimizer.zero_grad()
        outputs = model(X_batch).squeeze()
        loss = criterion(outputs, y_batch)
        loss.backward()
        optimizer.step()
        
        total_loss += loss.item() * len(y_batch)
        predictions = (outputs > 0.5).float()
        correct += (predictions == y_batch).sum().item()
        total += len(y_batch)
    
    return total_loss / total, correct / total


def evaluate(model, data_loader, criterion, device):
    """Evaluate model on a dataset."""
    model.eval()
    total_loss = 0
    correct = 0
    total = 0
    
    with torch.no_grad():
        for X_batch, y_batch in data_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            
            total_loss += loss.item() * len(y_batch)
            predictions = (outputs > 0.5).float()
            correct += (predictions == y_batch).sum().item()
            total += len(y_batch)
    
    return total_loss / total, correct / total


def train_model(model, train_loader, val_loader, device, epochs=10, lr=1e-3, patience=3, verbose=True):
    """Train model with early stopping."""
    model = model.to(device)
    optimizer = torch.optim.Adam(model.parameters(), lr=lr)
    criterion = nn.BCELoss()
    
    history = {
        'train_loss': [], 'train_acc': [],
        'val_loss': [], 'val_acc': []
    }
    
    best_val_loss = float('inf')
    patience_counter = 0
    best_state = None
    
    for epoch in range(epochs):
        train_loss, train_acc = train_epoch(model, train_loader, optimizer, criterion, device)
        val_loss, val_acc = evaluate(model, val_loader, criterion, device)
        
        history['train_loss'].append(train_loss)
        history['train_acc'].append(train_acc)
        history['val_loss'].append(val_loss)
        history['val_acc'].append(val_acc)
        
        if verbose:
            print(f"Epoch {epoch+1}/{epochs} - "
                  f"Loss: {train_loss:.4f} - Acc: {train_acc:.4f} - "
                  f"Val Loss: {val_loss:.4f} - Val Acc: {val_acc:.4f}")
        
        if val_loss < best_val_loss:
            best_val_loss = val_loss
            patience_counter = 0
            best_state = model.state_dict().copy()
        else:
            patience_counter += 1
            if patience_counter >= patience:
                if verbose:
                    print(f"Early stopping at epoch {epoch+1}")
                break
    
    if best_state is not None:
        model.load_state_dict(best_state)
    
    return history


def eval_binary_classification_torch(model, X_test, y_test, device, model_name=None, batch_size=64):
    """Evaluate PyTorch model and display metrics."""
    from sklearn.metrics import (
        confusion_matrix, classification_report,
        ConfusionMatrixDisplay, roc_curve, auc
    )
    
    model.eval()
    test_dataset = torch.utils.data.TensorDataset(X_test, y_test)
    test_loader = torch.utils.data.DataLoader(test_dataset, batch_size=batch_size)
    
    all_preds = []
    all_labels = []
    total_loss = 0
    criterion = nn.BCELoss()
    
    with torch.no_grad():
        for X_batch, y_batch in test_loader:
            X_batch, y_batch = X_batch.to(device), y_batch.to(device)
            outputs = model(X_batch).squeeze()
            loss = criterion(outputs, y_batch)
            total_loss += loss.item() * len(y_batch)
            all_preds.extend(outputs.cpu().numpy())
            all_labels.extend(y_batch.cpu().numpy())
    
    y_pred_proba = np.array(all_preds)
    y_true = np.array(all_labels)
    y_pred = (y_pred_proba >= 0.5).astype(int)
    
    test_loss = total_loss / len(y_test)
    test_acc = (y_pred == y_true).mean()
    
    print(f"\n{'='*60}")
    print(f"  {model_name or 'Model'} - Binary Classification Results")
    print(f"{'='*60}")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{'='*60}\n")
    
    class_names = ['Negative', 'Positive']
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    cm = confusion_matrix(y_true, y_pred)
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=axes[0], cmap='Blues', values_format='d')
    axes[0].set_title(f"{model_name or 'Model'} - Confusion Matrix")
    
    fpr, tpr, _ = roc_curve(y_true, y_pred_proba)
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


def plot_training_history(history, title=None):
    """Plot training history."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 4))
    
    epochs = range(1, len(history['train_loss']) + 1)
    
    axes[0].plot(epochs, history['train_loss'], label='Training Loss', linewidth=2)
    axes[0].plot(epochs, history['val_loss'], label='Validation Loss', linewidth=2)
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Loss')
    axes[0].set_title(f'{title + " - " if title else ""}Loss')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    axes[1].plot(epochs, history['train_acc'], label='Training Accuracy', linewidth=2)
    axes[1].plot(epochs, history['val_acc'], label='Validation Accuracy', linewidth=2)
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Accuracy')
    axes[1].set_title(f'{title + " - " if title else ""}Accuracy')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    plt.show()


def compare_models(results):
    """Compare multiple models visually."""
    names = list(results.keys())
    losses = [r[0] for r in results.values()]
    accuracies = [r[1] for r in results.values()]
    
    fig, axes = plt.subplots(1, 2, figsize=(12, 4))
    colors = plt.cm.Set2(np.linspace(0, 1, len(names)))
    
    bars1 = axes[0].bar(names, losses, color=colors)
    axes[0].set_ylabel('Loss')
    axes[0].set_title('Model Comparison - Test Loss')
    axes[0].set_ylim(0, max(losses) * 1.2)
    for bar, loss in zip(bars1, losses):
        axes[0].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.01,
                     f'{loss:.4f}', ha='center', va='bottom', fontsize=10)
    
    bars2 = axes[1].bar(names, [a * 100 for a in accuracies], color=colors)
    axes[1].set_ylabel('Accuracy (%)')
    axes[1].set_title('Model Comparison - Test Accuracy')
    axes[1].set_ylim(0, 100)
    for bar, acc in zip(bars2, accuracies):
        axes[1].text(bar.get_x() + bar.get_width()/2, bar.get_height() + 0.5,
                     f'{acc*100:.2f}%', ha='center', va='bottom', fontsize=10)
    
    plt.tight_layout()
    plt.show()
