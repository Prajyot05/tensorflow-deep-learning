import tensorflow as tf
import numpy as np

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