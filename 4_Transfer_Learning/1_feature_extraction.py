'''
What is Transfer Learning?
It is taking the patterns (also called weights) another model has learned from another problem and using them for our own problem.

There are two main benefits to using transfer learning:
  1. Can leverage an existing neural network architecture proven to work on problems similar to our own.
  2. Can leverage a working neural network architecture which has already learned patterns on similar data to our own.
     Then we can adapt those patterns to our own data.
'''

# To check if we are using a GPU
# !nvidia-smi

# Get data (10% of labels)
import zipfile

# Download data
# !wget https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip

# Unzip the downloaded file
zip_ref = zipfile.ZipFile("10_food_classes_10_percent.zip", "r")
zip_ref.extractall()
zip_ref.close()

# How many images in each folder?
import os

# Walk through 10 percent data directory and list number of files
for dirpath, dirnames, filenames in os.walk("10_food_classes_10_percent"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# Setup data inputs
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Usually hyperparameters are written in capital letters
IMAGE_SHAPE = (224, 224)
BATCH_SIZE = 32

train_dir = "10_food_classes_10_percent/train/"
test_dir = "10_food_classes_10_percent/test/"

train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

print("Training images:")
train_data_10_percent = train_datagen.flow_from_directory(train_dir,
                                               target_size=IMAGE_SHAPE,
                                               batch_size=BATCH_SIZE,
                                               class_mode="categorical")

print("Testing images:")
test_data = train_datagen.flow_from_directory(test_dir,
                                              target_size=IMAGE_SHAPE,
                                              batch_size=BATCH_SIZE,
                                              class_mode="categorical")

'''
Setting up callbacks (utitlies to call while our model trains)
Callbacks are extra functionality you can add to your models to be performed during or after training.

Some of the more popular callbacks are:
  1. Experiment tracking with TensorBoard - log the performance of multiple models and then view and compare these models
     in a visual way on TensorBoard (a dashboard for inspecting neural network parameters).
     Helpful to compare the results of different models on your data.

  2. Model checkpointing - save your model as it trains so you can stop training if needed and come back to continue off where you left.
     Helpful if training takes a long time and can't be done in one sitting.

  3. Early stopping - leave your model training for an arbitrary amount of time
     and have it stop training automatically when it ceases to improve.
     Helpful when you've got a large dataset and don't know how long training will take.
'''

'''
The TensorBoard callback
It's main functionality is saving a model's training performance metrics to a specified logging directory.
By default, logs are recorded every epoch using the update_freq='epoch' parameter.
This is a good default since tracking model performance too often can slow down model training.

We create a function for creating a TensorBoard callback because each model needs its own TensorBoard callback instance,
so the function will create a new one each time it's run.
'''
import datetime
import tensorflow as tf

def create_tensorboard_callback(dir_name, experiment_name):
    log_dir = dir_name + "/" + experiment_name + "/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
    tensorboard_callback = tf.keras.callbacks.TensorBoard(log_dir=log_dir)
    print(f"Saving TensorBoard log files to: {log_dir}")
    return tensorboard_callback