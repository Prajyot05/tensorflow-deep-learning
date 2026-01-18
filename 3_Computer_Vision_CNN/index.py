'''
What is a Convolutional Neural Network?
It a type of deep learning model specialized for processing grid-like data,
primarily images and videos, by automatically learning hierarchical patterns
from low-level features (edges, corners) to high-level concepts (objects, faces).

Architecture of a CNN:
    1. Input Image(s) -> Target images we'd like to discover patterns in
    2. Input Layer -> Takes in target images and preprocesses them for further layers
    3. Convolution Layer -> Extracts/learns the most important features from target images
    4. Hidden Activation -> Adds non-linearity to learned features (non-straight lines)
    5. Pooling Layer -> Reduces the dimensionality of learned image features
    6. Fully Connected Layer -> Further refines learned features from convolutional layers
    7. Output Layer -> Takes learned features and outputs them in shape of target labels
    8. Output Activation -> Adds non-linearities to output layer

A CNN is typically a stack of convolutional layers, pooling layers, and non-linear activations all jumbled together.
'''

'''
Computer vision is the practice of writing algorithms which can discover patterns in visual data.
'''

# Get the data (wget downloads the data into the current colab directory)
import zipfile
# wget https://storage.googleapis.com/ztm_tf_course/food_vision/pizza_steak.zip

# Unzip the downloaded file
zip_ref = zipfile.ZipFile("pizza_steak.zip")
zip_ref.extractall()
zip_ref.close()

# Inspect the data
'''
A very crucial step at the beginning of any machine learning project is to become one with the data.
For a computer vision project it usually means visualizing many samples of the data.
'''
import os
# Walk through the pizza_steak directory and list number of files
for dirpath, dirnames, filenames in os.walk("pizza_steak"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

# Another way to find out how many images are in a file
num_steak_images_train = len(os.listdir("pizza_steak/train/steak"))
print(num_steak_images_train)

# To visualize images first let's get the class names programmatically
import pathlib
import numpy as np
data_dir = pathlib.Path("pizza_steak/train/")
class_names = np.array(sorted([item.name for item in data_dir.glob('*')])) # Created a list of class names from the sub directories
print(class_names)

import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import random

def view_random_image(target_dir, target_class):
  # Setup target directory (we'll view images from here)
  target_folder = target_dir+target_class

  # Get a random image path
  random_image = random.sample(os.listdir(target_folder), 1)

  # Read in the image and plot it
  img = mpimg.imread(target_folder + "/" + random_image[0])
  plt.imshow(img)
  plt.title(target_class)
  plt.axis("off")

  print("Image Shape: ", img.shape)
  return img

img = view_random_image(target_dir="pizza_steak/train/", target_class="steak")

# The img returned is actuall an array (that can easily be converted into a tensor)
print(img)

# View the image shape
print(img.shape) # Returns the width, height and color channels

'''
Now we need a way to:
  1. Load our images
  2. Preprocess our images
  3. Build a CNN to find patterns in our images
  4. Compile our CNN
  5. Fit the CNN to our training data
'''
import tensorflow as tf
from tensorflow.keras.preprocessing.image import ImageDataGenerator

# Set the seed
tf.random.set_seed(42)

# Preprocess the data (Get all of the pixel values between 0 and 1, also called as scaling or normalization)
train_datagen = ImageDataGenerator(rescale=1./255) # Generates batches of tensor image data with real-time data augmentation
valid_datagen = ImageDataGenerator(rescale=1./255)

# Setup paths to our data directories
train_dir = "pizza_steak/train"
test_dir = "pizza_steak/test"

# Import data from directories and turn it into batches
train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               batch_size=32, # Yann LeCun famously advised, "Friends don't let friends use batch sizes greater than 32" 
                                               target_size=(224, 224), # We want all our images to be of this shape
                                               class_mode="binary", # Since we are importing our data in binary format
                                               seed=42) # For reproducibility

valid_data = valid_datagen.flow_from_directory(directory=test_dir,
                                               batch_size=32,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               seed=42)

# Build a CNN model (same as Tiny VGG on the CNN explainer website)
model_1 = tf.keras.Sequential([
  tf.keras.layers.Conv2D(filters=10,
                         kernel_size=3,
                         activation="relu",
                         input_shape=(224, 224, 3)),
  tf.keras.layers.Conv2D(10, 3, activation="relu"), # This layer is the exact same as the above one
  tf.keras.layers.MaxPool2D(pool_size=2,
                            padding="valid"),
  tf.keras.layers.Conv2D(10, 3, activation="relu"),
  tf.keras.layers.Conv2D(10, 3, activation="relu"),
  # tf.keras.layers.Activations(tf.nn.relu), We can use this instead of writing 'activation="relu"' as well
  tf.keras.layers.MaxPool2D(2),
  tf.keras.layers.Flatten(),
  tf.keras.layers.Dense(1, activation="sigmoid")
])

# Compile our CNN
model_1.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"]) # Generally a good metric for classification problems

# Fit the model
history_1 = model_1.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data), # If have 1500 images and batch size is 32, then per epoch images looked at will be 1500/32 = 47
                        validation_data=valid_data,
                        validation_steps=len(valid_data))

# It was taking over 100 seconds per epoch before switching the runtime type to T4 GPU, now it's taking 10 seconds on average
# The first epoch usually takes longer (by about 50%) than other epochs just because it needs to load the data into memory to run and find patterns on.

# Get the model summary
model_1.summary()

'''
Trying to see if the tensorflow playground model works on our image data.
'''

# 1. Create
model_2 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(4, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# 2. Compile
model_2.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# 3. Fit
history_2 = model_2.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))

# The performance of this model is terrible, it's getting a 50% accuracy at each epoch

model_2.summary()
# This model has 20 times more parameters than our CNN model, still it's performance is not good

# Improving the dense model

# 1. Create
model_3 = tf.keras.Sequential([
    tf.keras.layers.Flatten(input_shape=(224, 224, 3)),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(100, activation="relu"),
    tf.keras.layers.Dense(1, activation="sigmoid")
])

# 2. Compile
model_3.compile(loss="binary_crossentropy",
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])

# 3. Fit
history_3 = model_3.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=valid_data,
                        validation_steps=len(valid_data))

# This model got a better accuracy than before (75% by the end), but took 500 times more parameters than the CNN.

'''
Trainable parameters are patterns a model can learn from the data.
Intuitively it might seem as though the higher the number, the better it would be, and usually that is the case.
But in this case, the difference is in the two different styles of model we're using.
Where a series of dense layers has a number of different learnable parameters connected to each other
and hence a higher number of possible learnable patterns,
a CNN seeks to sort out and learn the most important patterns in an image.
So even if a CNN has lesser trainable parameters, they are often more helpful in deciphering between different features in an image.
'''

'''
Binary Classification Steps:
  1. Become one with the data (visualization)
  2. Preprocess the data (prepare it for the model, mainly through scaling/normalization, and turning our data into batches)
  3. Create the model (start with a baseline)
  4. Fit the model
  5. Evaluate the model
  6. Adjust different parameters and improve the model (try to beat our baseline)
  7. Repeat until satisfied (experimentation)
'''

# 1. Visualize the data
plt.figure()
plt.subplot(1, 2, 1)
steak_img = view_random_image("pizza_steak/train/", "steak")
plt.subplot(1, 2, 2)
pizza_img = view_random_image("pizza_steak/train/", "pizza")

# 2. Preprocess the data

# Define directory dataset paths
train_dir = "pizza_steak/train/"
test_dir = "pizza_steak/test/"

'''
Now we turn the data into batches (usually 32)
The reasons:
  1. All images might not fit into the memory at once.
  2. The model looking at them all in one go might not result in it finding the most accurate patterns
'''

# Create train and test data generators and rescale the data
from tensorflow.keras.preprocessing.image import ImageDataGenerator
train_datagen = ImageDataGenerator(rescale=1/255.) # Rescale is basically saying, when the images are loaded,divide all pixel values by 255
test_datagen = ImageDataGenerator(rescale=1/255.)

# Load in the image data from our directories and turn them into batches
train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               target_size=(224, 224),
                                               class_mode="binary",
                                               batch_size=32)

test_data = test_datagen.flow_from_directory(directory=test_dir,
                                             target_size=(224, 224),
                                             class_mode="binary",
                                             batch_size=32)

# Get a sample of the train data batch
images, labels = next(train_data) # Get the 'next' batch of image/labels from the train dataset
print(len(images), len(labels))

# How many batches are there? (1500 / 32)
len(train_data)

# Get the first two images
print(images[:2], images[0].shape)

# View the first batch of labels
print(labels)

'''
3. Create a CNN Model (start with a baseline)
A baseline is a relatively simple model or existing result that you setup when beginning a machine learning experiment.
And then as you keep experimenting, you try to beat the baseline.

Note: In deep learning, there is almost an infinite number of architectures you can create.
So one of the best ways to get started is to start with something simple and see if it works
and then introduce complexity as required.
E.g. look at which current model is performing the best in the field for your problem
'''

# Make the creating of our model a bit easier
from tensorflow.keras.optimizers import Adam
from tensorflow.keras.layers import Dense, Flatten, Conv2D, MaxPool2D, Activation
from tensorflow.keras import Sequential

# Create the model (this will be our baseline, a three layer convolutional neural network)
model_4 = Sequential([
    Conv2D(filters=10,
           kernel_size=3, # same as saying kernel_size=(3, 3)
           strides=1, # same as saying strides=(1, 1)
           padding="valid", # strides and padding have these values by default it's not necessary to set them
           activation="relu",
           input_shape=(224, 224, 3)), # Input layer
    Conv2D(10, 3, activation="relu"),
    Conv2D(10, 3, activation="relu"),
    Flatten(),
    Dense(1, activation="sigmoid") # Output layer (working with binary classification so only 1 ouput neuron)
])

# Compile the model
model_4.compile(loss="binary_crossentropy")

'''
Breakdown of the Conv2D layer:

  1. The "2D" means our inputs are two dimensional (height and width), even though they have 3 colour channels.
     The convolutions are run on each channel invididually.

  2. filters - These are the number of "feature extractors" that will be moving over our images.
               It decides how many filters should pass over an input tensor (sliding windows over an image).
               Higher values lead to more complex models.
               We don't define what these filters learn as they pass over an image, the neural network figures that out itself.

  3. kernel_size - It is the size/shape of our filters (sliding windows).
                   For example, a kernel_size of (3, 3) (or just 3) will mean each filter will have the size 3x3,
                   meaning it will look at a space of 3x3 pixels each time.
                   The smaller the kernel, the more fine-grained features it will extract.
                   So lower values learn smaller features, higher values learn larger features.

  4. padding - This pads the target tensor with zeroes (if 'same') to preserve the input shape.
               Or it leaves the target tensor as-it-is / unpadded (if 'valid'), lowering the output shape.
               If you want to keep more information in your input tensor, keep it 'same',
               If you want to compress the information passing through each layer, keep it 'valid'.

  5. stride - It is the number of steps a filter takes across an image at a time.
              E.g. A stride of 1 means the filter moves across each pixel 1 by 1. A stride of 2 means it moves 2 pixels at a time.
'''

print(model_4.summary())

# Check the lenghts of the training and test generators
print(len(train_data), len(test_data))

# Fit the model
history_4 = model_4.fit(train_data, # This is a combination of labels and sample data
                        epochs=5,
                        steps_per_epoch=len(train_data), # So that it goes through all 47 batches
                        validation_data=test_data,
                        validation_steps=len(test_data))

# Evaluating our model

import pandas as pd
pd.DataFrame(history_4.history).plot(figsize=(10, 7))

# Plot the validation and training curves seperately
def plot_loss_curves(history):
  loss = history.history["loss"]
  val_loss = history.history["val_loss"]

  accuracy = history.history["accuracy"]
  val_accuracy = history.history["val_accuracy"]

  epochs = range(len(history.history["loss"]))

  # Plot Loss
  plt.plot(epochs, loss, label="training_loss")
  plt.plot(epochs, val_loss, label="val_loss")
  plt.title("loss")
  plt.xlabel("epochs")
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label="training_accuracy")
  plt.plot(epochs, val_accuracy, label="val_accuracy")
  plt.title("accuracy")
  plt.xlabel("epochs")
  plt.legend()
  
plot_loss_curves(history_4)

'''
When a model's validation loss starts to increase, it's likely that the model is overfitting the training dataset.
This means, it's learning the patterns in the training dataset too well and thus the model's ability to generalize
to unseen data will be diminished.
'''