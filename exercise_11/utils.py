import keras
from datetime import datetime
import numpy as np
def train_model(model, X_train, y_train, model_name, epochs=50, early_stopping=True, patience=50, verbose = 1):
    
    log_dir = f"logs/{model_name}"
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
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
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        callbacks=callbacks,
        verbose=verbose
    )
    
    return model, history



def eval_model(model, X_test, y_test, model_name=None):
    """Evaluate model on test data and print summary."""
    from sklearn.metrics import r2_score
    
    test_loss, test_mae = model.evaluate(X_test, y_test, verbose=0)
    y_pred = model.predict(X_test, verbose=0)
    r2 = r2_score(y_test, y_pred)
    
    if model_name:
        print(f"\n{'='*50}")
        print(f"  {model_name} - Test Results")
        print(f"{'='*50}")
    else:
        print(f"\n{'='*50}")
        print(f"  Test Results")
        print(f"{'='*50}")
    
    print(f"  Test Loss (MSE): {test_loss:,.2f}")
    print(f"  Test MAE:        {test_mae:.2f}")
    print(f"  R² Score:        {r2:.4f} ({r2*100:.2f}% variance explained)")
    print(f"{'='*50}\n")
    
    return test_loss, test_mae, r2

def eval_classification(
    model: keras.Model, 
    X_test: np.ndarray, 
    y_test: np.ndarray, 
    class_names: list[str] = None,
    model_name: str = None
) -> tuple[float, float]:
    """Evaluate classification model and display confusion matrix."""
    import numpy as np
    import matplotlib.pyplot as plt
    from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay
    
    test_loss, test_acc = model.evaluate(X_test, y_test, verbose=0)
    y_pred = np.argmax(model.predict(X_test, verbose=0), axis=1)
    
    print(f"\n{'='*50}")
    print(f"  {model_name or 'Model'} - Classification Results")
    print(f"{'='*50}")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{'='*50}\n")
    
    print(classification_report(y_test, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_test, y_pred)
    fig, ax = plt.subplots(figsize=(10, 8))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(f"{model_name or 'Model'} - Confusion Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return test_loss, test_acc


def visualize_augmented_images(images, labels, class_names, n_samples=9):
    import matplotlib.pyplot as plt
    import numpy as np
    
    rows = 3
    cols = 3
    fig, axes = plt.subplots(rows, cols, figsize=(10, 10))
    axes = axes.ravel()
    
    indices = np.random.choice(len(images), n_samples, replace=False)
    
    for i, idx in enumerate(indices):
        img = images[idx]
        img = np.clip(img, 0, 1)
        axes[i].imshow(img)
        axes[i].set_title(class_names[labels[idx]], fontsize=10)
        axes[i].axis('off')
    
    plt.tight_layout()
    plt.show()


def start_tensorboard(logs_dir='logs', port=6006):
    """
    Display instructions for starting TensorBoard.
    
    Args:
        logs_dir: Directory containing TensorBoard logs (default: 'logs')
        port: Port to run TensorBoard on (default: 6006)
    """
    import os
    import socket
    
    # Get absolute path to logs
    logs_path = os.path.abspath(logs_dir)
    
    if not os.path.exists(logs_path):
        print(f"⚠️  Warning: Logs directory not found at {logs_path}")
        return
    
    # Get hostname and construct URL
    hostname = socket.gethostname()
    url = f"https://{hostname}-{port}.swedencentral.instances.azureml.ms"
    
    print("=" * 70)
    print("📊 TensorBoard Setup Instructions")
    print("=" * 70)
    print("\n1. Open a terminal in Azure ML Studio")
    print("\n2. Run this command:")
    print(f"\n   tensorboard --logdir {logs_path} --host 0.0.0.0 --port {port}\n")
    print("3. Open this URL in your browser:")
    print(f"\n   {url}\n")
    print("=" * 70)
    
    return url