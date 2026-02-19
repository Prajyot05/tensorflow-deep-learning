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

# Adjusting the model parameters
'''
Fitting a machine learning model comes in 3 steps:
  1. Create a baseline
  2. Beat the baseline by overfitting a larger model
  3. Reduce overfitting (also known as reguralization)

Ways to induce overfitting:
  1. Increase the number of conv layers
  2. Increase the number of conv filters
  3. Add another dense layer to the output of our flattened layer

Ways to reduce overfitting:
  1. Add data augmentation
  2. Add regularization layers (such as MaxPool2D)
  3. Add more data
'''

# Create the model (this is going to be our new baseline)
model_5 = Sequential([
    Conv2D(10, 3, activation="relu", input_shape=(224, 224, 3)),
    MaxPool2D(pool_size=2),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(1, activation="sigmoid")
])

# If a convolutional layer finds features in an image, max pooling finds the most important parts of those features.
# E.g. from every square four pixels, it chooses the max one, hence reducing the features to half.

model_5.compile(loss="binary_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

history_5 = model_5.fit(train_data,
                        epochs=5,
                        steps_per_epoch=len(train_data),
                        validation_data=test_data,
                        validation_steps=len(test_data))

model_5.summary()

plot_loss_curves(history_5)
# Ideally, training and validation loss curves should be similar to each other, max pool has helped us get closer to that.
# The curves looking similar means our model is performing just as well on the validation data as it is performing on the test data.

'''
Data Augmentation
It is the process of altering our training data, leading it to have more diversity and in turn allowing our models
to learn more generalizable (hopefully) patterns.
Altering might mean adjusting the rotation of an image, flipping it, cropping it, etc.
Hence thanks to data augmentation, we can train our models better without collecting more data.

Data augmentation is usually performed on the training data only.
When we use ImageDataGenerator and it's built-in data augmentation parameters, our images are left as they are in the directories
they are only modified as they're loaded into the model
'''

# Finding data augmentation

# Create ImageDataGenerator training instance with data augmentation
train_datagen_augmented = ImageDataGenerator(rescale=1/255.,
                                             rotation_range=0.2, # How much do you want to rotate the image?
                                             shear_range=0.2, # How much do you want to shear the image?
                                             zoom_range=0.2, # Zoom in randomly on an image
                                             width_shift_range=0.2, # Move the image around on the x-axis
                                             height_shift_range=0.3, # Move the image around on the y-axis
                                             horizontal_flip=True) # Do you want to flip an image?

# Create ImageDataGenerator without data augmentation
train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)


# Import data and augment it form training directory
print("Augmented training data")
train_data_augmented = train_datagen_augmented.flow_from_directory(directory=train_dir,
                                                              target_size=(224, 224),
                                                              batch_size=32,
                                                              class_mode="binary",
                                                              shuffle=False)

# Create non-augmented train data batches
print("Non-augmented training data")
train_data = train_datagen.flow_from_directory(directory=train_dir,
                                               target_size=(224, 224),
                                               batch_size=32,
                                               class_mode="binary",
                                               shuffle=False)

# Create non-augmented test data batches
print("Non-augmented test data")
test_data = test_datagen.flow_from_directory(directory=test_dir,
                                             target_size=(224, 224),
                                             batch_size=32,
                                             class_mode="binary")

# Visualize sample batches of augmented and non-augmented data
# Note: Labels aren't augmented, only data (images)
images, labels = next(train_data)
augmented_images, augmented_labels = next(train_data_augmented)

# Show original and augmented image
import random
random_number = random.randint(0, 32) # Since our batch size is 32
plt.imshow(images[random_number])
plt.title(f"Original image for {labels[random_number]}")
plt.axis(False)
plt.figure()
plt.imshow(augmented_images[random_number])
plt.title(f"Augmented image for {augmented_labels[random_number]}")
plt.axis(False)

'''
Notice how some of the augmented images look like slightly warped versions of the original image.
This means our model will be forced to try and learn patterns in less-than-perfect images,
which is often the case when using real-world images.
'''

# Training our model on the augmented data
model_6 = Sequential([
  Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
  MaxPool2D(pool_size=2), # reduce number of features by half
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Flatten(),
  Dense(1, activation='sigmoid')
])

# Compile the model
model_6.compile(loss='binary_crossentropy',
                optimizer=Adam(),
                metrics=['accuracy'])

# Fit the model
history_6 = model_6.fit(train_data_augmented,
                        epochs=5,
                        steps_per_epoch=len(train_data_augmented),
                        validation_data=test_data,
                        validation_steps=len(test_data))

# Check model's performance
plot_loss_curves(history_6)

# Shuffling the data
train_data_augmented_shuffled = train_datagen_augmented.flow_from_directory(directory=train_dir,
                                                              target_size=(224, 224),
                                                              batch_size=32,
                                                              class_mode="binary",
                                                              shuffle=True)

model_7 = Sequential([
    Conv2D(10, 3, activation="relu", input_shape=(224, 224, 3)),
    MaxPool2D(), # pool_size=2 by default
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Conv2D(10, 3, activation="relu"),
    MaxPool2D(),
    Flatten(),
    Dense(1, activation="sigmoid")
])

model_7.compile(loss="binary_crossentropy",
                optimizer=Adam(),
                metrics=["accuracy"])

model_7.fit(train_data_augmented_shuffled,
            epochs=5,
            steps_per_epoch=len(train_data_augmented_shuffled),
            validation_data=test_data,
            validation_steps=len(test_data))

plot_loss_curves(history_7)

'''
Now that we've beaten our baseline, here's how we can improve our model further:
  1. Increase the number of Conv2D and MaxPool2D layers
  2. Increase the number of filters in each convolutional layer
  3. Train for longer (more epochs)
  4. Find an ideal learning rate
  5. Get more data
  6. Use transfer learning to leverage what another image model has learned
     and adjust it for our use case
'''

# Making a prediction with our trained model on our own custom data
# View our example image
# !wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/images/03-steak.jpeg 
steak = mpimg.imread("03-steak.jpeg")
plt.imshow(steak)
plt.axis(False)

print(steak.shape)
# The shape is not compatible with our model
# Since our model takes in images of shapes (224, 224, 3), we've got to reshape our custom image to use it with our model.

# Create a function to import an image and resize it to be able to be used with our model
def load_and_prep_image(filename, img_shape=224):
  """
  Reads an image from filename, turns it into a tensor
  and reshapes it to (img_shape, img_shape, colour_channel).
  """

  # Read in target file
  img = tf.io.read_file(filename)

  # Decode the read file into a tensor
  img = tf.image.decode_image(img, channels=3) # (our model is trained on images with 3 colour channels and sometimes images have 4 colour channels)

  # Resize the image (to the same size our model was trained on)
  img = tf.image.resize(img, size = [img_shape, img_shape])

  # Rescale the image (get all values between 0 and 1)
  img = img/255.
  return img

# Load in and preprocess our custom image
steak = load_and_prep_image("03-steak.jpeg")
print(steak)

'''
Although our image is in the same shape as the images our model has been trained on, we're still missing a dimension.
Our model was trained in batches, so the batch size becomes the first dimension.
So in reality, our model was trained on data in the shape of (batch_size, 224, 224, 3).
We can fix this by adding an extra to our custom image tensor using tf.expand_dims.
'''
pred = model_7.predict(tf.expand_dims(steak, axis=0))
# This gives us the prediction probability (how likely the image is to belong to one class or the other)

# Now let's try to visualize the image as well as the model's prediction
print(class_names)

# We can index the predicted class by rounding the prediction probability
pred_class = class_names[int(tf.round(pred)[0][0])]
print(pred_class)

def pred_and_plot(model, filename, class_names):
  '''
  Imports an image located at filename, makes a prediction on it with
  a trained model and plots the image with the predicted class as the title.
  '''
  # Import the target image and preprocess it
  img = load_and_prep_image(filename)

  # Make a prediction
  pred = model.predict(tf.expand_dims(img, axis=0))

  # Get the predicted class
  pred_class = class_names[int(tf.round(pred)[0][0])]

  # Plot the image and predicted class
  plt.imshow(img)
  plt.title(f"Prediction: {pred_class}")
  plt.axis(False)

# Testing our function
pred_and_plot(model_7, "03-steak.jpeg", class_names)

'''
Multi-class Image Classification

The steps remain the same:
  1. Become one with the data
  2. Preprocess the data (prepare it for a model)
  3. Create a model (start with a baseline)
  4. Fit the model (overfit it to make sure it works)
     Overfitting is generally a good thing because it shows us that our model is learning something,
     and it is generally very easy to get rid of.
  5. Evaluate the model
  6. Adjust different parameters and improve model (try to beat your baseline)
  7. Repeat until satisfied
'''

# 1. Getting the data
import zipfile

# Download zip file of 10_food_classes images
# !wget https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip 

# Unzip the downloaded file
zip_ref = zipfile.ZipFile("10_food_classes_all_data.zip", "r")
zip_ref.extractall()
zip_ref.close()

import os

# Walk through the directory
for dirpath, dirnames, filenames in os.walk("10_food_classes_all_data"):
  print(f"There are {len(dirnames)} directories and {len(filenames)} images in '{dirpath}'.")

  import pathlib
import numpy as np

train_dir = "10_food_classes_all_data/train/"
test_dir = "10_food_classes_all_data/test/"

# Get the class names for our multi-class dataset (without using tensorflow)
data_dir = pathlib.Path(train_dir)
class_names = np.array(sorted([item.name for item in data_dir.glob('*')]))
print(class_names)

import random
img = view_random_image(target_dir=train_dir,
                        target_class=random.choice(class_names))

# 2. Preprocess the data

# Rescale the data and create data generator instances
train_datagen = ImageDataGenerator(rescale=1/255.)
test_datagen = ImageDataGenerator(rescale=1/255.)

# Load data in from directories and turn it into batches
train_data = train_datagen.flow_from_directory(train_dir,
                                               target_size=(224, 224),
                                               batch_size=32,
                                               class_mode='categorical') # Since we're dealing with more than 2 classes

test_data = train_datagen.flow_from_directory(test_dir,
                                              target_size=(224, 224),
                                              batch_size=32,
                                              class_mode='categorical')

# 3. Create a model

model_8 = Sequential([
  Conv2D(10, 3, activation='relu', input_shape=(224, 224, 3)),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Conv2D(10, 3, activation='relu'),
  Conv2D(10, 3, activation='relu'),
  MaxPool2D(),
  Flatten(),
  Dense(10, activation='softmax') # Changed to have 10 neurons (same as number of classes) and 'softmax' activation
])

# Compile the model
model_8.compile(loss="categorical_crossentropy", # Changed from 'binary_crossentropy' due to multi-class classification
                optimizer=tf.keras.optimizers.Adam(),
                metrics=["accuracy"])