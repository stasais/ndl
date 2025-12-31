# Solution Assignment 4
**Buinitskii Stanislav**

## Exercise 1: Extended Random Search

### Part a) Extending the Hyperparameter Search

I added 3 new hyperparameters to the search: **learning rate**, **optimizer type**, and **batch size**.

#### New Hyperparameters

**1. Learning Rate**
- Range: 1e-4 to 1e-2 (log scale)
- Why: This is probably the most important hyperparameter. Too high and training becomes unstable, too low and it takes forever to converge. I sampled in log scale because the effect of learning rate is multiplicative.
- Expected impact: Should have the biggest effect on results.

**2. Optimizer**
- Choices: adam, sgd, rmsprop
- Why: Different optimizers can work better for different problems. Adam is usually a safe default, but SGD sometimes generalizes better, and RMSprop is somewhere in between.
- Expected impact: SGD typically needs more careful tuning of learning rate, so I expected it might perform worse with random search.

**3. Batch Size**
- Choices: 32, 64, 128
- Why: Affects the noise in gradient estimates. Smaller batches = more noise = can help escape local minima but slower training. Larger batches = faster but might overfit.
- Expected impact: Probably not as big as learning rate, but worth exploring.

#### Results Summary

Ran 100 configurations with 20 epochs each. Best accuracy: **88.7%** with:
- hidden_layers: [256, 512, 64, 512]
- dropout: 0.033
- learning_rate: 0.000194
- optimizer: rmsprop
- batch_size: 32

Looking at the results:
- **Adam and RMSprop** clearly outperformed SGD. Most SGD runs ended up in the bottom half of the ranking.
- **Lower learning rates** (around 1e-4 to 1e-3) worked better than higher ones.
- **Batch size** didn't seem to matter that much, all three values appeared in top results.
- Deeper networks with 3-4 layers did slightly better than shallow ones.

The worst results were mostly SGD with very low learning rates - these probably didn't converge in 20 epochs.

### Part b) Fixing the Data Leak

#### The Problem

Looking at `train_model()` in utils.py, I noticed it uses `validation_split=0.2`. The issue is that Keras just takes the last 20% of the array - it doesn't shuffle.

Then in `run_search()`, after training, the code does:
```python
_, val_acc = model.evaluate(X_train[:10000], y_train[:10000], verbose=0)
```

This is wrong because we're checking accuracy on the first 10k samples of X_train, but the model was trained on roughly the first 48k samples. So we're measuring training accuracy, not validation accuracy.

#### The Fix

1. Added `train_test_split()` at the beginning to create X_train, X_val, y_train, y_val explicitly
2. Changed `train_model()` to require `validation_data` parameter
3. Fixed `run_search()` to evaluate on X_val, y_val

Now all models train on the same training set and get evaluated on the same held-out validation set.

---

## Exercise 2: Keras Tuner - Hyperband

### Chosen Strategy: Hyperband

I went with Hyperband because it's easier to understand and more efficient for this kind of search.

#### How Hyperband Works

Hyperband is based on "successive halving". The basic idea:
1. Start a bunch of random configurations with a small budget (e.g., 5 epochs each)
2. Evaluate them all, then kill the worst half
3. Give the survivors more budget (e.g., 10 epochs) and repeat
4. Keep going until one configuration remains

It's basically random search but smarter about where to spend compute time. If a model is clearly bad after a few epochs, why train it for 20 more?

**Reference:** Li et al., "Hyperband: A Novel Bandit-Based Approach to Hyperparameter Optimization", JMLR 2018

#### Comparison to Random Search

| Aspect | Random Search | Hyperband |
|--------|--------------|-----------|
| Exploration | Random | Random |
| Resource allocation | Fixed (all configs get same epochs) | Adaptive (good configs get more) |
| Efficiency | Wastes time on bad configs | Early stopping of bad configs |

### Implementation

Used Keras Tuner with the same search space as Exercise 1:
- num_layers: 1-4
- units per layer: [64, 128, 256, 512]
- dropout: 0.0-0.4
- learning_rate: 1e-4 to 1e-2 (log scale)
- optimizer: adam, sgd, rmsprop

Ran both Hyperband and RandomSearch tuners for comparison.

### Results

- **Hyperband:** test acc = 0.8588, time = 334s
  - Best config: 3 layers [64, 256, 512], dropout=0.142, lr=0.000358, rmsprop
  
- **Random Search:** test acc = 0.8765, time = 624s
  - Best config: 3 layers, dropout=0.254, lr=0.000408, adam

Interestingly, Random Search found a better model but took almost twice as long. Hyperband was faster because it stopped bad configurations early, but in this case it seems like some configs that looked bad early actually would have been good with more training.

### Is Hyperband inherently better than random search with manual tuning?

Not necessarily - Hyperband is efficient at exploring a defined search space automatically, but an experienced practitioner who knows which hyperparameters matter most can often get similar results faster by narrowing the search space and tuning a few key parameters manually.

---

## LLM Usage

Throughout both exercises, I used LLM assistants (GitHub Copilot, Grok, Claude Opus 4.5) for code scaffolding, debugging, and help with documentation. The core logic and analysis decisions were mine, but the LLMs helped speed up implementation and catch errors.
