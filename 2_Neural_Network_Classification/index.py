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

# Improving the model
# 1. Create
model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(100), # Add 100 dense neurons
    tf.keras.layers.Dense(10), # Add another layer with 10 neurons
    tf.keras.layers.Dense(1)
])

# 2. Compile
model_2.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                optimizer = tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# 3. Fit
model_2.fit(X, y, epochs=100, verbose=0)

# 4. Evaluate
model_2.evaluate(X, y)

# Visualize the predictions against the actual data
import numpy as np
'''
This function will:
 1. Take input features (X) and labels(y)
 2. Create a meshgrid of different X values
 3. Make predictions across the meshgrid
 4. Plot the predictions as well as a line between different zones (where each unique class falls)
'''

# Plots the decision boundary created by a model predicting on X                    
def plot_decision_boundary(model, X, y):
  # Define the axis boundaries of the plot and create a meshgrid.
  x_min, x_max = X[:, 0].min() - 0.1, X[:, 0].max() + 0.1 # 0.1 is for margin
  y_min, y_max = X[:, 1].min() - 0.1, X[:, 1].max() + 0.1

  xx, yy = np.meshgrid(np.linspace(x_min, x_max, 100), np.linspace(y_min, y_max, 100))
  
  # Create X values that we are going to make predictions on
  x_in = np.c_[xx.ravel(), yy.ravel()] # Stack 2D arrays together

  # Make prediction
  y_pred = model.predict(x_in)
  
  # Check for multi-class
  if(len(y_pred[0])) > 1:
    print("Doing multiclass classification")
    # We have to reshape our predictions to get them ready for plotting
    y_pred = np.argmax(y_pred, axis=1).reshape(xx.shape)
  else:
    print("Doing binary classification")
    y_pred = np.round(y_pred).reshape(xx.shape)
    
  # Plot the decision boundary
  plt.contourf(xx, yy, y_pred, cmap=plt.cm.RdYlBu, alpha=0.7)
  plt.scatter(X[:, 0], X[:, 1], c=y, s=40, cmap = plt.cm.RdYlBu)
  plt.xlim(xx.min(), xx.max())
  plt.ylim(yy.min(), yy.max())

# Plotting the model
plot_decision_boundary(model_2, X, y)

'''
Activation Function: 
  A mathematical function applied to the output of a neuron.
  It decides whether a neuron should be activated based on the weighted sum of inputs and a bias term.
  It introduces non-linearity, enabling the model to learn and represent complex data patterns.
  Without it, even a deep neural network would behave like a simple linear regression model.

Non-linear functions allow neural networks to form curved decision boundaries, making them capable of handling complex patterns.
'''

# Creating a model with non-linear activation functions
# 1. Create
model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"), 
    tf.keras.layers.Dense(1, activation="sigmoid") # Three layers with 4, 4, 1 neuron for respective layer
])

# 2. Compile
model_3.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                optimizer = tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# 3. Fit
history = model_3.fit(X, y, epochs=100, verbose=0)

# 4. Evaluate
model_3.evaluate(X, y)

# Plot
plot_decision_boundary(model_3, X, y)

# Create a temporary tensor to test how different activation functions change it
A = tf.cast(tf.range(-10, 10), dtype=tf.float32)

# Linear - Does not change the input at all
tf.keras.activations.linear(A)

# Sigmoid - y = 1 / (1 + tf.exp(-x))
tf.keras.activations.sigmoid(A)

# ReLu - y = tf.maximum(0, x)
tf.keras.activations.relu(A)

# Seperating training and test datasets
X_train, y_train = X[:800], y[:800]
X_test, y_test = X[800:], y[800:]

# 1. Create
model_4 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# 2. Compile
model_4.compile(loss = tf.keras.losses.BinaryCrossentropy(), # loss function tells how wrong the patterns being formed are
                optimizer = tf.keras.optimizers.Adam(learning_rate=0.01), # optimizer tells how the model should be improved, learning rate tells how much the model should be improved
                metrics = ["accuracy"]) # Lower learning rate means lesser changes to the patterns for improvement
'''
Example: if learning rate is 0.001 then for each epoch, the weights will be improved by a scale of 0.001
So by making learning rate 0.01, we have increased the potential of the model to improve its weight 10 times as much.
That's not exactly how it works, but it's a good way to think about it for developing intuition
'''

# 3. Fit
history = model_3.fit(X_train, y_train, epochs = 25, verbose = 0) # The losses change much faster compared to lower learning rate models

# 4. Evaluate
model_4.evaluate(X_test, y_test)

# Plot the decision boundary for the training and test sets
plt.figure(figsize=(12, 6))
plt.subplot(1, 2, 1)
plt.title("Train")
plot_decision_boundary(model_4, X_train, y_train)

plt.subplot(1, 2, 2)
plt.title("Test")
plot_decision_boundary(model_4, X_test, y_test)

plt.show()

# Plot the loss (or training) curves

# history.history is a record of training loss values and metrics values at successive epochs, as well as validation loss values and validation metrics values (if applicable).
pd.DataFrame(history.history).plot()
plt.title("model_4 loss curves")
plt.show()

'''
Finding the best learning rate
The best learning rate would be the one where the loss decreases the most during training
Use these to find the best learning rate:
  1. A learning rate callback (callback is like an extra piece of functionality that you can add to the model WHILE it is training)
  2. Another model
  3. A modified loss curves plot
'''

# 1. Create
model_5 = tf.keras.Sequential([
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# 2. Compile
model_5.compile(loss = tf.keras.losses.BinaryCrossentropy(),
                optimizer = tf.keras.optimizers.Adam(),
                metrics = ["accuracy"])

# 3. Create a learning rate callback
lr_scheduler = tf.keras.callbacks.LearningRateScheduler(lambda epoch: 1e-4 * 10**(epoch/20))
# What the above line means is that for every epoch to traverse a set of learning rate values starting from 1e-4 and increasing by 10**(epoch/20) every epoch

# r. Fitting the model
history_5 = model_5.fit(X_train, y_train, epochs=100, callbacks=[lr_scheduler])

pd.DataFrame(history_5.history).plot(figsize=(10, 7), xlabel="epochs")
plt.show()
# The ideal learning rate is somewhere where the loss is 10 times smaller than where it is the lowest

# Plot the learning rate vs the loss
lrs = 1e-4 * (10 ** (tf.range(100) / 20)) # Because we did 100 epochs
plt.figure(figsize=(10, 7))
plt.semilogx(lrs, history_5.history["loss"])
plt.xlabel("Learning Rate")
plt.ylabel("Loss")
plt.title("Learning Rate vs Loss")
plt.show()

'''
Different classification evaluation methods:
  1. Accuracy
  2. Precision
  3. Recall
  4. F1-Score - Based on precision and recall, usually a very good metric
  5. Confusion Matrix
  6. Classification Report (from scikit-learn)
'''

# Check the accuracy of our model
loss, accuracy = model_5.evaluate(X_test, y_test)
print(f"Model loss on the test set: {loss}")
print(f"Model accuracy on the test set: {(accuracy*100):.2f}%")

# Creating a Confusion Matrix
from sklearn.metrics import confusion_matrix

# Make predictions
y_preds = model_5.predict(X_test) # This is currently in prediction probability form, we need to convert it to binary (0 or 1)
y_preds = tf.round(y_preds) # Rounds it to 0 or 1

print(confusion_matrix(y_test, y_preds))

# Making the confusion matrix prettier
import itertools
figsize = (10, 10)

# Create the confusion matrix
cm = confusion_matrix(y_test, y_preds)
cm_norm = cm.astype("float") / cm.sum(axis=1)[:, np.newaxis] # Normalize our confusion matrix
print(cm_norm)

n_classes = cm.shape[0] # Right now we have 2 classes but in the future we would have more

# To prettify it
fig, ax = plt.subplots(figsize=figsize)
# Create matrix plot
cax = ax.matshow(cm, cmap=plt.cm.Blues)
fig.colorbar(cax)

# Create classes
classes = False # For if we have a list of classes

if classes:
  lables = classes
else:
  labels = np.arange(cm.shape[0])

# Label the axes
ax.set(title="Confusion Matrix",
       xlabel="Predicted Label",
       ylabel="True Label",
       xticks=np.arange(n_classes),
       yticks=np.arange(n_classes),
       xticklabels=labels,
       yticklabels=labels)

# Set x-axis labels to bottom
ax.xaxis.set_label_position("bottom")
ax.xaxis.tick_bottom()

# Adjust label size
ax.xaxis.label.set_size(20)
ax.yaxis.label.set_size(20)
ax.title.set_size(20)

# Set the threshold for different colors
threshold = (cm.max() + cm.min()) / 2.0 # This will give our confusion matrix different shades of squares depending on how many values are in there

# Plot the text on each cell
for i, j in itertools.product(range(cm.shape[0]), range(cm.shape[1])):
  plt.text(j, i, f"{cm[i, j]} ({cm_norm[i, j] * 100:.1f}%)",
           horizontalalignment="center",
           color="white" if cm[i, j] > threshold else "black",
           size=15)
  
'''
Multi-class Classification: When you have more than 2 classes
We will be using the tensorflow fashion dataset that has 9 classes
'''
from tensorflow.keras.datasets import fashion_mnist

# The data has already been sorted into training and test dataset
(train_data, train_labels), (test_data, test_labels) = fashion_mnist.load_data()

# Check the shape of a single example
print(train_data[0].shape)
print(train_labels[0].shape)

# Plot a single sample
plt.imshow(train_data[0])

# Checkout the sample's label
print(train_labels[0])