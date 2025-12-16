'''
What are classification problems?
Teaching a model to sort items into distinct groups

Type of classification problems:
1. Binary classification (one thing or another)
2. Multi-class classification (more than one thing or another)
3. Multi-label classification (multiple label options per sample)
'''

from sklearn.datasets import make_circles
import pandas as pd
import matplotlib.pyplot as plt
import tensorflow as tf

# Make 1000 samples
n_samples = 1000

# Create circles
X, y = make_circles(n_samples, noise=0.03, random_state=42)

# Visualize the circles
circles = pd.DataFrame({"X0": X[:, 0], "X1": X[:, 1], "label": y})
plt.scatter(X[:, 0], X[:, 1], c = y, cmap = plt.cm.RdYlBu)

tf.random.set_seed(42)

# Creating the model
model_1 = tf.keras.Sequential([
    tf.keras.layers.Dense(1),
    tf.keras.layers.Dense(1)
])

model_1.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                optimizer = tf.keras.optimizers.SGD(),
                metrics=["accuracy"])

model_1.fit(X, y, epochs=100, verbose=0)

# Evaluating the model
model_1.evaluate(X, y) # Terrible accuracy at this point