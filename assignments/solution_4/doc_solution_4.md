# Solution Assignment 4
**Buinitskii Stanislav**

## Part b) Fixing the Data Leak (1 point)

### The Problem

The original `run_search()` implementation has two data leak issues:

1. **Deterministic validation split**: `train_model()` uses `validation_split=0.2` which takes the **last 20%** of data without shuffling. All models see the exact same validation set.

2. **Evaluating on training data**: After training, the code evaluates with:
   ```python
   _, val_acc = model.evaluate(X_train[:10000], y_train[:10000], verbose=0)
   ```
   This evaluates on the **first 10,000 samples of training data**, not on validation data!

### Why This Is Wrong

- The model trains on indices 0-47,999 and validates on indices 48,000-59,999
- But we measure accuracy on indices 0-10,000 — **which the model was trained on**
- This gives inflated accuracy scores that don't reflect generalization ability

### The Fix

1. **`utils.py`**: Changed `validation_data` to a required parameter (no fallback to `validation_split`)

2. **Notebook**: 
   - Split data explicitly at the beginning using `train_test_split()` with shuffling
   - Pass `validation_data=(X_val, y_val)` to `train_model()`
   - Evaluate on `X_val, y_val` instead of `X_train[:10000]`
