import keras
import numpy as np
import matplotlib.pyplot as plt
import matplotlib as mpl
import tensorflow as tf
from sklearn.metrics import confusion_matrix, classification_report, ConfusionMatrixDisplay


def train_model(model, X_train, y_train, model_name, epochs=50, early_stopping=True, patience=10, verbose=1, batch_size=32):
    """Train a model with TensorBoard logging and optional early stopping."""
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
        batch_size=batch_size,
        callbacks=callbacks,
        verbose=verbose
    )
    
    return model, history


def train_model_ds(model, train_ds, val_ds, model_name, epochs=50, early_stopping=True, patience=10, verbose=1):
    """Train a model using tf.data.Dataset with TensorBoard logging and optional early stopping."""
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
        train_ds,
        validation_data=val_ds,
        epochs=epochs,
        callbacks=callbacks,
        verbose=verbose
    )
    
    return model, history


def eval_classification(model, X_test, y_test, class_names=None, model_name=None, save_path=None):
    """Evaluate classification model and display confusion matrix."""
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
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(f"{model_name or 'Model'} - Confusion Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
    
    return test_loss, test_acc


def eval_classification_ds(model, test_ds, class_names=None, model_name=None):
    """Evaluate classification model using a tf.data.Dataset."""
    test_loss, test_acc = model.evaluate(test_ds, verbose=0)
    
    y_true = []
    y_pred_all = []
    for images, labels in test_ds:
        preds = model.predict(images, verbose=0)
        y_pred_all.append(np.argmax(preds, axis=1))
        y_true.append(labels.numpy())
    
    y_true = np.concatenate(y_true)
    y_pred = np.concatenate(y_pred_all)
    
    print(f"\n{'='*50}")
    print(f"  {model_name or 'Model'} - Classification Results")
    print(f"{'='*50}")
    print(f"  Test Loss:     {test_loss:.4f}")
    print(f"  Test Accuracy: {test_acc:.4f} ({test_acc*100:.2f}%)")
    print(f"{'='*50}\n")
    
    print(classification_report(y_true, y_pred, target_names=class_names))
    
    cm = confusion_matrix(y_true, y_pred)
    fig, ax = plt.subplots(figsize=(8, 6))
    disp = ConfusionMatrixDisplay(confusion_matrix=cm, display_labels=class_names)
    disp.plot(ax=ax, cmap='Blues', values_format='d')
    plt.title(f"{model_name or 'Model'} - Confusion Matrix")
    plt.xticks(rotation=45, ha='right')
    plt.tight_layout()
    plt.show()
    
    return test_loss, test_acc


def plot_training_history(histories, names, save_path=None):
    """Plot training and validation accuracy/loss for multiple models."""
    fig, axes = plt.subplots(1, 2, figsize=(14, 5))
    
    for history, name in zip(histories, names):
        axes[0].plot(history.history['accuracy'], label=f'{name} train')
        axes[0].plot(history.history['val_accuracy'], '--', label=f'{name} val')
    axes[0].set_title('Model Accuracy')
    axes[0].set_xlabel('Epoch')
    axes[0].set_ylabel('Accuracy')
    axes[0].legend()
    axes[0].grid(True, alpha=0.3)
    
    for history, name in zip(histories, names):
        axes[1].plot(history.history['loss'], label=f'{name} train')
        axes[1].plot(history.history['val_loss'], '--', label=f'{name} val')
    axes[1].set_title('Model Loss')
    axes[1].set_xlabel('Epoch')
    axes[1].set_ylabel('Loss')
    axes[1].legend()
    axes[1].grid(True, alpha=0.3)
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()


# --- GradCAM (adapted from exercise 11) ---

def make_gradcam_heatmap(img_array, model, base_model, last_conv_layer_name, pred_index=None):
    """Generate GradCAM heatmap. Supports both nested (transfer learning) and flat (custom) models."""
    is_nested = base_model is not None and any(layer.name == base_model.name for layer in model.layers)
    
    if is_nested:
        target_layer = base_model.get_layer(last_conv_layer_name)
        base_grad_model = keras.Model(
            base_model.inputs,
            [target_layer.output, base_model.output]
        )
        
        classifier_layers = []
        found_base = False
        for layer in model.layers:
            if layer.name == base_model.name:
                found_base = True
                continue
            if found_base:
                classifier_layers.append(layer)
        
        classifier_input = keras.Input(shape=base_model.output.shape[1:])
        x = classifier_input
        for layer in classifier_layers:
            x = layer(x)
        classifier_model = keras.Model(classifier_input, x)
        
        with tf.GradientTape() as tape:
            conv_output, base_output = base_grad_model(img_array)
            tape.watch(conv_output)
            preds = classifier_model(base_output)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_output)
    else:
        grad_model = keras.models.Model(
            model.inputs, 
            [model.get_layer(last_conv_layer_name).output, model.output]
        )
        
        with tf.GradientTape() as tape:
            conv_output, preds = grad_model(img_array)
            if pred_index is None:
                pred_index = tf.argmax(preds[0])
            class_channel = preds[:, pred_index]

        grads = tape.gradient(class_channel, conv_output)

    pooled_grads = tf.reduce_mean(grads, axis=(0, 1, 2))
    conv_output = conv_output[0]
    heatmap = conv_output @ pooled_grads[..., tf.newaxis]
    heatmap = tf.squeeze(heatmap)
    heatmap = tf.maximum(heatmap, 0) / tf.math.reduce_max(heatmap)
    
    return heatmap.numpy()


def create_gradcam_visualization(img, heatmap, alpha=0.4, background_intensity=0.5):
    """Overlay GradCAM heatmap on image."""
    heatmap_resized = np.uint8(255 * heatmap)
    jet = mpl.colormaps["jet"]
    jet_colors = jet(np.arange(256))[:, :3]
    jet_heatmap = jet_colors[heatmap_resized]

    jet_heatmap = keras.utils.array_to_img(jet_heatmap)
    jet_heatmap = jet_heatmap.resize((img.shape[1], img.shape[0]))
    jet_heatmap = keras.utils.img_to_array(jet_heatmap)

    superimposed_img = jet_heatmap * alpha + img * background_intensity
    superimposed_img = keras.utils.array_to_img(superimposed_img)
    
    return superimposed_img


def show_gradcam_grid(images, labels_true, labels_pred, class_names, model, base_model, last_conv_layer_name, preprocess_fn=None, save_path=None):
    """Show a grid of original images, heatmaps, and overlays."""
    n = len(images)
    fig, axes = plt.subplots(3, n, figsize=(4 * n, 12))
    
    for col in range(n):
        img = images[col]
        img_input = img.copy()
        if preprocess_fn is not None:
            img_input = preprocess_fn(img_input)
        img_array = np.expand_dims(img_input, axis=0)
        
        heatmap = make_gradcam_heatmap(img_array, model, base_model, last_conv_layer_name)
        superimposed = create_gradcam_visualization(img, heatmap)
        
        # display image scaled to [0, 255] uint8
        display_img = img
        if display_img.max() <= 1.0:
            display_img = (display_img * 255).astype(np.uint8)
        
        axes[0, col].imshow(display_img)
        axes[0, col].set_title(f"True: {class_names[labels_true[col]]}", fontsize=10)
        axes[0, col].axis('off')
        
        axes[1, col].imshow(heatmap, cmap='jet')
        axes[1, col].set_title("GradCAM", fontsize=10)
        axes[1, col].axis('off')
        
        axes[2, col].imshow(superimposed)
        axes[2, col].set_title(f"Pred: {class_names[labels_pred[col]]}", fontsize=10)
        axes[2, col].axis('off')
    
    plt.tight_layout()
    if save_path:
        plt.savefig(save_path, dpi=150, bbox_inches='tight')
        plt.close()
    else:
        plt.show()
