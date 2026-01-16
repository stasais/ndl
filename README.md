# LectureNotes

Repository for lecture notes, scripts and notebooks for the NDL3 Deep Learning course.

## Structure

Each exercise is its own self-contained folder with all necessary notebooks, utility scripts, and data files.

```
LectureNotes/
├── exercise_8/          # Keras fundamentals
├── exercise_9/          # Image classification with FCNNs, Hyperparameter search
├── exercise_10/         # Convolutional Neural Networks (CNNs)
├── exercise_11/         # Transfer Learning & GradCAM
├── exercise_12/         # Recurrent Neural Networks (LSTMs)
├── assignments/         # Graded assignments
└── pyproject.toml       # Environment configuration
```

## Environment

The Python environment is managed with [uv](https://docs.astral.sh/uv/), but usage is optional. You can also use pip, poetry, or any other package manager of your choice.

**With uv:**

```bash
uv sync
```

## Exercises

| Exercise        | Topic                      | Description                                                                                                                                   |
| --------------- | -------------------------- | --------------------------------------------------------------------------------------------------------------------------------------------- |
| **Exercise 8**  | Keras Fundamentals         | Introduction to Keras, activation functions (Sigmoid, ReLU, Leaky ReLU), dropout, batch normalization, skip connections, Keras Functional API |
| **Exercise 9**  | Image Classification       | Fully Connected Neural Networks (FCNNs) for Fashion MNIST, limitations of dense networks for images, random hyperparameter search             |
| **Exercise 10** | CNNs                       | Convolutional Neural Networks for CIFAR-10, convolutional layers, pooling operations, hierarchical feature learning                           |
| **Exercise 11** | Transfer Learning          | Pretrained models, fine-tuning, GradCAM visualization for model interpretability                                                              |
| **Exercise 12** | Recurrent Neural Networks  | LSTM networks for IMDB sentiment classification, word embeddings, GloVe pretrained embeddings                                                 |

## Requirements

- Keras 3.x with TensorFlow backend (or PyTorch/JAX)
- See `pyproject.toml` for full dependency list
