import tensorflow as tf
import numpy as np
import matplotlib.pyplot as plt
import pandas as pd
from tensorflow.keras.utils import plot_model

tf.random.set_seed(42)

X = tf.range(-100, 100, 4)

y = X + 10

model = tf.keras.Sequential([
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1)
])

model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.Adam(learning_rate = 0.0001),
              metrics=["mae"])

model.fit(tf.expand_dims(X, axis=-1), y, epochs=100)

y_pred = model.predict(np.array([16.0]))
print(y_pred)

''' 
Common ways to improve a model (changing hyperparameters):
    1. Changing the learning rate
    2. Fitting on more data and for longer duration
    3. Changing the optimization function
    4. Adding layers
    5. Increasing the number of hidden units
    6. Changing the activation function

    Parameters - Patterns a neural network learns, these are not coded by us
    Hyperparameters - It is like a dial that we can adjust to tweak the performance of the neural network
'''

# Split the data into train and test sets
X_train = X[:40] # The first 40 elements of X (80% of the data)
y_train = y[:40]

X_test = X[40:] # The last 10 elements (20% of the data)
y_test = y[40:]

print(len(X_train), len(y_train), len(X_test), len(y_test))

# Visualizing the data
plt.figure(figsize=(10, 7))

# Plot training data in blue
plt.scatter(X_train, y_train, c="b", label="Training Data")

# Plot test data in green
plt.scatter(X_test, y_test, c="g", label="Test Data")

# Show the legend
plt.legend()

# Building a neural network for this data

# 1. Create the model, this time we manually give it the input shape
model = tf.keras.Sequential([
    tf.keras.layers.Dense(1, input_shape=(1,)) # Just one number being passed
])

# 2. Compile the model
model.compile(loss=tf.keras.losses.mae,
              optimizer=tf.keras.optimizers.SGD(),
              metrics=["mae"])

# Get summary
model.summary()
''' 
OUTPUT:
Model: "sequential_2"
┏━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━━━━━━━━━━┳━━━━━━━━━━━━━━━┓
┃ Layer (type)                    ┃ Output Shape           ┃       Param # ┃
┡━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━━━━━━━━━━╇━━━━━━━━━━━━━━━┩
│ dense_5 (Dense)                 │ (None, 1)              │             2 │
└─────────────────────────────────┴────────────────────────┴───────────────┘
 Total params: 2 (8.00 B) -> Total number of parameters in the model
 Trainable params: 2 (8.00 B) -> The parameters (patterns) that the model can update as it trains
 Non-trainable params: 0 (0.00 B) -> The parameters that the model cannot change (usually used when bringing in already trained parameters from different models using transfer learning)
'''

# 3. Fit the model
model.fit(X_train, y_train, epochs=100, verbose=0)

# Another way to visualize

plot_model(model=model, show_shapes=True)

# Visualizing the model's predictions (plotting them against ground truth labels)
y_preds = model.predict(X_test)

# Plotting function
def plot_predictions(train_data=X_train,
                     train_labels=y_train,
                     test_data=X_test,
                     test_labels=y_test,
                     predictions=y_preds):
  plt.figure(figsize=(10, 7))

  # Plot training data in blue
  plt.scatter(train_data, train_labels, c="b", label="Training Data")

  # Plot test data in green
  plt.scatter(test_data, test_labels, c="g", label="Testing Data")

  # Plot model's prediction in red
  plt.scatter(test_data, predictions, c="r", label="Predictions")

  # Show the legend
  plt.legend()

plot_predictions()

# Evaluating our model's predictions with regression evaluation metrics

# Evaluating on the test
model.evaluate(X_test, y_test)

# Calculating the mean absolute error (great starter metric)
mae_1 = tf.keras.losses.MAE(y_true=y_test, y_pred=tf.squeeze(y_preds)) # Compressing y_preds to match the shape of y_test
print(mae_1)

# Calculating mean square error (useful when larger errors are more significant than smaller errors)
mse_1 = tf.keras.losses.MSE(y_true=y_test, y_pred=tf.squeeze(y_preds))
print(mse_1)

# Experimeting using variation in model training

model_2 = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

model_2.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["mse"])

model_2.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100)

y_preds_2 = model_2.predict(X_test)
plot_predictions(predictions=y_preds_2)

mae_2 = tf.keras.losses.MAE(y_true=y_test, y_pred=tf.squeeze(y_preds_2))
print(mae_2)
mse_2 = tf.keras.losses.MSE(y_true=y_test, y_pred=tf.squeeze(y_preds_2))
print(mse_2)

# Trying another model variation
model_3 = tf.keras.Sequential([
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1)
])

model_3.compile(loss=tf.keras.losses.mae,
                optimizer=tf.keras.optimizers.SGD(),
                metrics=["mae"])

model_3.fit(tf.expand_dims(X_train, axis=-1), y_train, epochs=100)

y_preds_3 = model_3.predict(X_test)
plot_predictions(predictions=y_preds_3) # This time the model did overfitting, it learned the training data too well and thats not good when testing new data

mae_3 = tf.keras.losses.MAE(y_true=y_test, y_pred=tf.squeeze(y_preds_3)) # Compressing y_preds to match the shape of y_test
print(mae_3)
mse_3 = tf.keras.losses.MSE(y_true=y_test, y_pred=tf.squeeze(y_preds_3))
print(mse_3)

# Comparing the results of our model's variations

model_results = [
    ["model_1", mae_1.numpy(), mse_2.numpy()],
    ["model_2", mae_2.numpy(), mse_2.numpy()],
    ["model_3", mae_3.numpy(), mse_3.numpy()]
]

all_results = pd.DataFrame(model_results, columns=["model", "mae", "mse"])
print(all_results) # So now we know what works and what doesn't

# Saving the model (HDF5 is deprecated, keras is recommended)
model_3.save("best_model_till_now.keras")

# Loading the saved model
loaded_model = tf.keras.models.load_model("best_model_till_now.keras")
print(loaded_model.summary())

# Working with larger datasets
import tensorflow as tf
import pandas as pd
import matplotlib.pyplot as plt

# Read the insurance dataset
insurance = pd.read_csv("https://raw.githubusercontent.com/stedy/Machine-Learning-with-R-datasets/refs/heads/master/insurance.csv")

# We need to convert the non-numerical data into numerical data before passing it to a model (numerical encoding)
# This function turns categorical data into one-hot encoding format
insurance_one_hot = pd.get_dummies(insurance)
print(insurance_one_hot)

# Create features and labels
X = insurance_one_hot.drop("charges", axis=1)
y = insurance_one_hot["charges"]

# Creating training and test sets
from sklearn.model_selection import train_test_split

# Randomly shuffles the data (we've told it to give 20% data to test and rest to train)
X_train, X_test, y_train, y_test = train_test_split(X, y, test_size=0.2, random_state=42)

# Now build a neural network to learn from that data
tf.random.set_seed(42)

# 1. Create
insurance_model = tf.keras.Sequential([
    tf.keras.layers.Dense(100),
    tf.keras.layers.Dense(10),
    tf.keras.layers.Dense(1),
])

# 2. Compile
insurance_model.compile(loss=tf.keras.losses.mae,
                        optimizer=tf.keras.optimizers.Adam(),
                        metrics=["mae"])

# 3. Fit
history = insurance_model.fit(X_train, y_train, epochs=200, verbose=0)

# Evaluate the results of the insurance_model on the test data
insurance_model.evaluate(X_test, y_test)

# Plot history (also known as loss curve or training curve), shows the progresss of loss reduction during training
pd.DataFrame(history.history).plot()
plt.ylabel("loss")
plt.xlabel("epochs")

# How long should you train for?
# Tensoflow has a solution to this problem called EarlyStopping Callback
# It is a component we can add to our model to stop training once the model stops improving a certain metric