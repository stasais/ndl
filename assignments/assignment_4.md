# Assignment 4: Advanced Hyperparameter Tuning

## Exercise 1: Extended Random Search (7 points)

### Part a) Extending the Hyperparameter Search (6 points)

Based on the notebook `mnist_fashion_fcnn_simple.ipynb` from exercise 9, extend the hyperparameter search with additional parameters and implement a systematic search strategy. **Feel free to adapt/add/delete code in any way you see fit.**

**Tasks:**

1. **Extend the search space** with at least 3 additional hyperparameters beyond `hidden_layers` and `dropout_rate`.

2. **Modify the necessary functions:**

   - Update `create_hyperparams()` to generate configurations for your new parameters
   - Adapt `create_fcnn()` to accept and use these parameters
   - Modify `run_search()` to pass the parameters correctly

3. **Run at least 100 different configurations** and document your findings.

4. **Explain your choices:** For each hyperparameter you add, explicitly describe:
   - Why you chose this parameter
   - What range/values you selected and why
   - What impact you expect it to have

**Note:** You will need to modify multiple functions, but you can base everything on the existing `mnist_fashion_fcnn_simple.ipynb` notebook structure.

### Part b) Fixing the Data Leak (1 point)

There is a subtle data leak in the current search implementation.

**Hint:** Look carefully at how we perform validation in the `train_model()` function in `utils.py`. Consider what data the model sees during training and how this might affect the fairness of comparing different random seeds or configurations.

Identify the issue, explain what's wrong, and fix it.

---

## Exercise 2: Keras Tuner (4 points)

Explore systematic hyperparameter optimization using Keras Tuner. The principle behind it is quite similar to the implementation we did by hand, however it has support for other search algorithms than random search. Documentation can be found under: https://keras.io/keras_tuner/.

**Tasks:**

1. **Choose and explain one optimization strategy:**

   - Either Hyperband Optimization
   - Or Bayesian Optimization

   Briefly describe how the chosen method works (search for an appropriate reference paper or other academic resource! Hint: look at what keras-tuner cites) and compare to random search.

2. **Implement the search:**

   - Use Keras Tuner with your chosen strategy on the Fashion MNIST dataset
   - Build a small comparison experiment with random search from exercise 1. (e.g. convergence speed.)
   - Is such an approach inherently better than random search with additional manual tuning? Reason in one sentence.

**Deliverables:** A notebook or script+markdown demonstrating your implementation with clear explanations and a summary of your findings. Note any use of an LLM in detail please.

---

## Submission

Submission via Moodle.

**Total Points: 11**
