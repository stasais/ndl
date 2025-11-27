import keras
from datetime import datetime

def train_model(model, X_train, y_train, model_name, epochs=50):

    
    log_dir = f"logs/{model_name}"
    tensorboard_callback = keras.callbacks.TensorBoard(log_dir=log_dir, histogram_freq=1)
    
    history = model.fit(
        X_train, y_train,
        validation_split=0.2,
        epochs=epochs,
        batch_size=32,
        callbacks=[tensorboard_callback],
        verbose=1
    )
    
    return history


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