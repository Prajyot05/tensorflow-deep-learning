'''
In fine-tuning transfer learning the pre-trained model weights from another model are unfrozen
and tweaked during to better suit your own data.
'''

# Check if we're using a GPU
# !nvidia-smi

# Creating helper functions
# Get helper_functions.py script from pre-built Github page
# !wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py

# Import helper functions we're going to use
from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, walk_through_dir

# Get 10% of the data of the 10 classes
# !wget https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_10_percent.zip

unzip_data("10_food_classes_10_percent.zip")

# Walk through 10 percent data directory and list number of files
walk_through_dir("10_food_classes_10_percent")

# Create training and test directories
train_dir = "10_food_classes_10_percent/train/"
test_dir = "10_food_classes_10_percent/test/"

'''
Now we've got some image data, we need a way of loading it into a TensorFlow compatible format.
Previously, we've used the ImageDataGenerator class. But it is deprecated,
so we will be using tf.keras.prepreprocessing.image_dataset_from_directory().

One of the main benefits of using tf.keras.prepreprocessing.image_dataset_from_directory() rather than
ImageDataGenerator is that it creates a tf.data.Dataset object rather than a generator.
The main advantage of this is the tf.data.Dataset API is much more efficient (faster)
than the ImageDataGenerator API which is paramount for larger datasets.
'''

# Create data inputs
import tensorflow as tf

IMG_SIZE = (224, 224) # define image size

train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=train_dir,
                                                                            image_size=IMG_SIZE,
                                                                            label_mode="categorical", # multi-class
                                                                            batch_size=32)

test_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(directory=test_dir,
                                                                           image_size=IMG_SIZE,
                                                                           label_mode="categorical")

'''
For now, the main parameters we're concerned about in the image_dataset_from_directory() funtion are:

directory - the filepath of the target directory we're loading images in from.
image_size - the target size of the images we're going to load in (height, width).
batch_size - the batch size of the images we're going to load in.
             For example if the batch_size is 32 (the default), batches of 32 images and labels at a time will be passed to the model.
'''

# Check out the class names of our dataset
# Notice how the image arrays come out as tensors of pixel values where as the labels come out as one-hot encodings.
train_data_10_percent.class_names

# See an example batch of data
for images, labels in train_data_10_percent.take(1):
  print(images, labels)

'''
Building a transfer learning model using the Keras Functional API

We're going to go through the following steps:

  1. Instantiate a pre-trained base model object by choosing a target model such as EfficientNetV2B0
     from tf.keras.applications.efficientnet_v2, setting the include_top parameter to False
     (we do this because we're going to create our own top, which are the output layers for the model).

  2. Set the base model's trainable attribute to False to freeze all of the weights in the pre-trained model.

  3. Define an input layer for our model, for example, what shape of data should our model expect?

  4. [Optional] Normalize the inputs to our model if it requires. Some computer vision models such as ResNetV250 require their
     inputs to be between 0 & 1.

  5. Pass the inputs to the base model.

  6. Pool the outputs of the base model into a shape compatible with the output activation layer
     (turn base model output tensors into same shape as label tensors).
     This can be done using tf.keras.layers.GlobalAveragePooling2D() or tf.keras.layers.GlobalMaxPooling2D()
     though the former is more common in practice.

  7. Create an output activation layer using tf.keras.layers.Dense() with the appropriate activation function and number of neurons.

  8. Combine the inputs and outputs layer into a model using tf.keras.Model().

  9. Compile the model using the appropriate loss function and choose of optimizer.
  
  10. Fit the model for desired number of epochs and with necessary callbacks (in our case, we'll start off with the TensorBoard callback).
'''

# 1. Create base model with tf.keras.applications
# We're not including top layer because we want to add our own dense layer on top
# This is because by default it has 1000 layers (due to it being trained on ImageNet) but we want 10 (= number of classes)
base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)

# 2. Freeze the base model (so the pre-learned patterns remain)
base_model.trainable = False

# 3. Create inputs into the base model
inputs = tf.keras.layers.Input(shape=(224, 224, 3), name="input_layer")

# 4. Pass the inputs to the base_model (note: using tf.keras.applications, EfficientNetV2 inputs don't have to be normalized)
x = base_model(inputs)
# Check data shape after passing it to base_model
print(f"Shape after base_model: {x.shape}")

# 6. Average pool the outputs of the base model (aggregate all the most important information, reduce number of computations)
x = tf.keras.layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
print(f"After GlobalAveragePooling2D(): {x.shape}")

# 7. Create the output activation layer
outputs = tf.keras.layers.Dense(10, activation="softmax", name="output_layer")(x)

# 8. Combine the inputs with the outputs into a model
model_0 = tf.keras.Model(inputs, outputs)

# 9. Compile the model
model_0.compile(loss='categorical_crossentropy',
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

# 10. Fit the model (we use less steps for validation so it's faster)
history_0 = model_0.fit(train_data_10_percent,
                                 epochs=5,
                                 steps_per_epoch=len(train_data_10_percent),
                                 validation_data=test_data_10_percent,
                                 validation_steps=int(0.25 * len(test_data_10_percent)), # Go through 25% of the validation data so epochs are faster
                                 callbacks=[create_tensorboard_callback("transfer_learning", "10_percent_feature_extract")]) # Track our model's training logs for visualization later

'''
The kind of transfer learning we used above is called feature extraction transfer learning,
We passed our custom data to an already pre-trained model (EfficientNetV2B0), asked it "what patterns do you see?"
and then put our own output layer on top to make sure the outputs were tailored to our desired number of classes.
'''

# Check layers in our base model
for layer_number, layer in enumerate(base_model.layers):
  print(layer_number, layer.name)

base_model.summary()
# It would have been very time-consuming to manually create so many layers ourselves, that is why transfer learning is preffered.
model_0.summary()
plot_loss_curves(history_0)

'''
Getting a feature vector from a trained model

What does the tf.keras.layers.GlobalAveragePooling2D() layer do?
It transforms a 4D tensor into a 2D tensor by averaging the values across the inner-axes.
'''

# Define input tensor shape (same number of dimensions as the output of efficientnetv2-b0)
input_shape = (1, 4, 4, 3)

# Create a random tensor
tf.random.set_seed(42)
input_tensor = tf.random.normal(input_shape)
print(f"Random input tensor:\n {input_tensor}\n")

# Pass the random tensor through a global average pooling 2D layer
global_average_pooled_tensor = tf.keras.layers.GlobalAveragePooling2D()(input_tensor)
print(f"2D global average pooled random tensor:\n {global_average_pooled_tensor}\n")

# Check the shapes of the different tensors
print(f"Shape of input tensor: {input_tensor.shape}")
print(f"Shape of 2D global averaged pooled input tensor: {global_average_pooled_tensor.shape}")

# This is the same as GlobalAveragePooling2D()
tf.reduce_mean(input_tensor, axis=[1, 2]) # average across the middle axes

'''
Doing this not only makes the output of the base model compatible with the input shape
requirement of our output layer (tf.keras.layers.Dense()), it also condenses the information
found by the base model into a lower dimension feature vector.
'''

'''
What is a feature vector?
It is a learned representation of the input data
(a compressed form of the input data based on how the model sees it)
'''

'''
Running different transfer learning experiments:
  1. Model 1: Use feature extraction transfer learning on 1% of the training data with data augmentation.
  2. Model 2: Use feature extraction transfer learning on 10% of the training data with data augmentation
     and save the results to a checkpoint.
  3. Model 3: Fine-tune the Model 2 checkpoint on 10% of the training data with data augmentation.
  4. Model 4: Fine-tune the Model 2 checkpoint on 100% of the training data with data augmentation.
'''

# Download and unzip data
# !wget https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_1_percent.zip
unzip_data("10_food_classes_1_percent.zip")

# Create training and test dirs
train_dir_1_percent = "10_food_classes_1_percent/train/"
test_dir = "10_food_classes_1_percent/test/"

walk_through_dir("10_food_classes_1_percent")

IMG_SIZE = (224, 224)
train_data_1_percent = tf.keras.preprocessing.image_dataset_from_directory(train_dir_1_percent,
                                                                           label_mode="categorical",
                                                                           batch_size=32, # default
                                                                           image_size=IMG_SIZE)
test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                label_mode="categorical",
                                                                image_size=IMG_SIZE)

'''
Using Data Augmentation while feeding the data to the model using preprocessing layers
Adding a data augmentation layer to the model has the following benefits:

  1. Preprocessing of the images (augmenting them) happens on the GPU rather than on the CPU (much faster).
  2. Images are best preprocessed on the GPU where as text and structured data are more suited to be preprocessed on the CPU.
  3. Image data augmentation only happens during training so we can still export our whole model and use it elsewhere.
     And if someone else wanted to train the same model as us, including the same kind of data augmentation, they could.
'''
import tensorflow as tf
from tensorflow import keras
from tensorflow.keras import layers

data_augmentation = keras.Sequential([
  keras.Input(shape=(224, 224, 3)),
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.2),
  layers.RandomHeight(0.2),
  layers.RandomWidth(0.2),
  # No need to to Rescaling for EfficientNetV2B0 as it has it built-in
], name ="data_augmentation")

# View a random image
import matplotlib.pyplot as plt
import matplotlib.image as mpimg
import os
import random

target_class = random.choice(train_data_1_percent.class_names) # choose a random class
target_dir = "10_food_classes_1_percent/train/" + target_class # create the target directory
random_image = random.choice(os.listdir(target_dir)) # choose a random image from target directory
random_image_path = target_dir + "/" + random_image # create the choosen random image path
img = mpimg.imread(random_image_path) # read in the chosen target image
plt.imshow(img) # plot the target image
plt.title(f"Original random image from class: {target_class}")
plt.axis(False); # turn off the axes

# Augment the image
augmented_img = data_augmentation(tf.expand_dims(img, axis=0)) # data augmentation model requires shape (None, height, width, 3)
plt.figure()
plt.imshow(tf.squeeze(augmented_img)/255.) # requires normalization after augmentation
plt.title(f"Augmented random image from class: {target_class}")
plt.axis(False)

# Feature extraction transfer learning
# Setup input shape and base model, freezing the base model layers
input_shape = (224, 224, 3)
base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)
base_model.trainable = False

# Create input layer
inputs = layers.Input(shape=input_shape, name="input_layer")

# Add in data augmentation Sequential model as a layer
x = data_augmentation(inputs)

# Give base_model inputs (after augmentation) and don't train it
x = base_model(x, training=False)

# Pool output features of base model
x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)

# Put a dense layer on as the output
outputs = layers.Dense(10, activation="softmax", name="output_layer")(x)

# Make a model with inputs and outputs
model_1 = keras.Model(inputs, outputs)

# Compile the model
model_1.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(),
              metrics=["accuracy"])

# Fit the model
history_1_percent = model_1.fit(train_data_1_percent,
                    epochs=5,
                    steps_per_epoch=len(train_data_1_percent),
                    validation_data=test_data,
                    validation_steps=int(0.25* len(test_data)), # validate for less steps
                    # Track model training logs
                    callbacks=[create_tensorboard_callback("transfer_learning", "1_percent_data_aug")])

# Check out model summary
model_1.summary()

# Evaluate on the test data
results_1_percent_data_aug = model_1.evaluate(test_data)
results_1_percent_data_aug

# How does the model go with a data augmentation layer with 1% of data
plot_loss_curves(history_1_percent)

# Model 2: Feature extraction transfer learning with 10% of data and data augmentation

train_dir_10_percent = "10_food_classes_10_percent/train/"
test_dir = "10_food_classes_10_percent/test/"

# Setup data inputs
import tensorflow as tf

IMG_SIZE = (224, 224)
train_data_10_percent = tf.keras.preprocessing.image_dataset_from_directory(train_dir_10_percent,
                                                                            label_mode="categorical",
                                                                            image_size=IMG_SIZE)
# Note: the test data is the same as the previous experiment, we could
test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                label_mode="categorical",
                                                                image_size=IMG_SIZE)

# Create model 2 with data augmentation built-in
from tensorflow import keras
from keras import layers
from keras.models import Sequential

data_augmentation = keras.Sequential([
  layers.RandomFlip("horizontal"),
  layers.RandomRotation(0.2),
  layers.RandomZoom(0.2), # This handles both height and width zooming now!
  # preprocessing.Rescaling(1./255) # keep for ResNet50V2, remove for EfficientNet
], name ="data_augmentation")

# Setup the input shape to our model
input_shape = (224, 224, 3)

# Create a frozen base model (also called the backbone)
base_model = tf.keras.applications.efficientnet_v2.EfficientNetV2B0(include_top=False)
base_model.trainable = False

# Create input and output layers (including the layers in between)
inputs = layers.Input(shape=input_shape, name="input_layer") # create input layer
x = data_augmentation(inputs) # augment our training images
x = base_model(x, training=False) # pass augmented images to base model but keep it in inference mode, so batchnorm layers don't get updated
x = layers.GlobalAveragePooling2D(name="global_average_pooling_layer")(x)
outputs = layers.Dense(10, activation="softmax", name="output_layer")(x)
model_2 = tf.keras.Model(inputs, outputs)

# Compile
model_2.compile(loss="categorical_crossentropy",
              optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), # use Adam optimizer with base learning rate
              metrics=["accuracy"])

'''
The ModelCheckpoint callback
It gives you the ability to save your model,
as a whole in the SavedModel format or the weights (patterns) only to a specified directory as it trains.

What's the difference between saving the entire model (SavedModel format) and saving the weights only?
The SavedModel format saves a model's architecture, weights and training configuration all in one folder.
It makes it very easy to reload your model exactly how it is elsewhere. However, if you do not want to share all
of these details with others, you may want to save and share the weights only
(these will just be large tensors of non-human interpretable numbers).
If disk space is an issue, saving the weights only is faster and takes up less space than saving the whole model.
'''

# Setup checkpoint path
checkpoint_path = "ten_percent_model_checkpoints_weights/checkpoint.weights.h5"

# Create a ModelCheckpoint callback that saves the model's weights only
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(filepath=checkpoint_path,
                                                         save_weights_only=True, # set to False to save the entire model
                                                         save_best_only=True, # save only the best model weights instead of a model every epoch
                                                         save_freq="epoch", # save every epoch
                                                         verbose=1)

# Fit the model saving checkpoints every epoch
initial_epochs = 5
history_10_percent_data_aug = model_2.fit(train_data_10_percent,
                                          epochs=initial_epochs,
                                          validation_data=test_data,
                                          validation_steps=int(0.25 * len(test_data)), # do less steps per validation (quicker)
                                          callbacks=[create_tensorboard_callback("transfer_learning", "10_percent_data_aug"),
                                                     checkpoint_callback])

# Evaluate on the test data
results_10_percent_data_aug = model_2.evaluate(test_data)
results_10_percent_data_aug

# Loading in checkpointed weights
model_2.load_weights(checkpoint_path)
loaded_weights_model_results = model_2.evaluate(test_data)

# If the results from our native model and the loaded weights are the same, this should output True
# It will not return True because of the way computers store tiny decimal numbers, but they are still very close
results_10_percent_data_aug == loaded_weights_model_results

import numpy as np
# Check to see if loaded model results are very close to native model results
np.isclose(np.array(results_10_percent_data_aug), np.array(loaded_weights_model_results))

'''
Model 3: Fine-tuning an existing model on 10% of the data

High-level example of fine-tuning an EfficientNet model:
  Bottom layers (layers closer to the input data) stay frozen
  where as top layers (layers closer to the output data) are updated during training.

Uptil now all of the layers in the base model (EfficientNetV2B0) were frozen during training.
Now we're going to switch to fine-tuning transfer learning. This means we'll be using the same base model except we'll be unfreezing
some of its layers (ones closest to the top) and running the model for a few more epochs.

The idea with fine-tuning is to start customizing the pre-trained model more to our own data.

Note: Fine-tuning usually works best after training a feature extraction model for a few epochs and with large amounts of data.
'''

# Layers in loaded model
model_2.layers

for layer_number, layer in enumerate(model_2.layers):
  print(f"Layer number: {layer_number} | Layer name: {layer.name} | Layer type: {layer} | Trainable? {layer.trainable}")
     

model_2.summary()

# Access the base_model layers of model_2
model_2_base_model = model_2.layers[2]
model_2_base_model.name
     
# How many layers are trainable in our model_2_base_model?
print(len(model_2_base_model.trainable_variables)) # layer at index 2 is the EfficientNetV2B0 layer (the base model)

# Check which layers are tuneable (trainable)
for layer_number, layer in enumerate(model_2_base_model.layers):
  print(layer_number, layer.name, layer.trainable)

'''
Now to fine-tune the base model to our own data, we're going to unfreeze the top 10 layers
and continue training our model for another 5 epochs.

How many layers should you unfreeze when training?
Generally, the less data you have, the less layers you want to unfreeze and the more gradually you want to fine-tune.
'''

# Make all the layers in model_2_base_model trainable
model_2_base_model.trainable = True

# Freeze all layers except for the last 10
for layer in model_2_base_model.layers[:-10]:
  layer.trainable = False

# Recompile the whole model (always recompile after any adjustments to a model)
model_2.compile(loss="categorical_crossentropy",
                optimizer=tf.keras.optimizers.Adam(learning_rate=0.0001), # lr is 10x lower than before for fine-tuning (rule of thumb)
                metrics=["accuracy"])

# Check which layers are tuneable (trainable)
for layer_number, layer in enumerate(model_2_base_model.layers):
  print(layer_number, layer.name, layer.trainable)

# How many trainable variables do we have now?
print(len(model_2.trainable_variables))
# The model has a total of 12 trainable variables, the last 10 layers of the base model and the weight and bias parameters
# of the Dense output layer.

# Fine tune for another 5 epochs
fine_tune_epochs = initial_epochs + 5

# Refit the model (same as model_2 except with more trainable layers)
history_fine_10_percent_data_aug = model_2.fit(train_data_10_percent,
                                               epochs=fine_tune_epochs,
                                               validation_data=test_data,
                                               initial_epoch=history_10_percent_data_aug.epoch[-1], # start from previous last epoch
                                               validation_steps=int(0.25 * len(test_data)),
                                               callbacks=[create_tensorboard_callback("transfer_learning", "10_percent_fine_tune_last_10")])

'''
Note: Fine-tuning usually takes far longer per epoch than feature extraction (due to updating more weights throughout a network).
'''

# Evaluate the model on the test data
results_fine_tune_10_percent = model_2.evaluate(test_data)

# Creating a function to compare training histories
def compare_historys(original_history, new_history, initial_epochs=5):
  """
  Compares two model history objects.
  """
  # Get original history measurements
  acc = original_history.history["accuracy"]
  loss = original_history.history["loss"]
  
  val_acc = original_history.history["val_accuracy"]
  val_loss = original_history.history["val_loss"]

  # Combine original history with new history
  total_acc = acc + new_history.history["accuracy"]
  total_loss = loss + new_history.history["loss"]

  total_val_acc = val_acc + new_history.history["val_accuracy"]
  total_val_loss = val_loss + new_history.history["val_loss"]

  # Make plots for Accuracy and Loss
  plt.figure(figsize=(8, 8))
  plt.subplot(2, 1, 1)
  plt.plot(total_acc, label='Training Accuracy')
  plt.plot(total_val_acc, label='Validation Accuracy')
  plt.plot([initial_epochs-1, initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
  plt.legend(loc='lower right')
  plt.title('Training and Validation Accuracy')

  plt.subplot(2, 1, 2)
  plt.plot(total_loss, label='Training Loss')
  plt.plot(total_val_loss, label='Validation Loss')
  plt.plot([initial_epochs-1, initial_epochs-1],
            plt.ylim(), label='Start Fine Tuning') # reshift plot around epochs
  plt.legend(loc='upper right')
  plt.title('Training and Validation Loss')
  plt.xlabel('epoch')
  plt.show()

# Compare the histories of our feature extraction model and fine-tuned model
compare_historys(history_10_percent_data_aug, history_fine_10_percent_data_aug, initial_epochs=5)

'''
Model 4: Fine-tuning an existing model all of the data
'''

# Download and unzip 10 classes of data with all images
# !wget https://storage.googleapis.com/ztm_tf_course/food_vision/10_food_classes_all_data.zip
unzip_data("10_food_classes_all_data.zip")

# Setup data directories
train_dir = "10_food_classes_all_data/train/"
test_dir = "10_food_classes_all_data/test/"

# How many images are we working with now?
walk_through_dir("10_food_classes_all_data")

# Turn the images into tensors datasets.
import tensorflow as tf
IMG_SIZE = (224, 224)
train_data_10_classes_full = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                                 label_mode="categorical",
                                                                                 image_size=IMG_SIZE)

# Note: this is the same test dataset we've been using for the previous modelling experiments
test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                label_mode="categorical",
                                                                image_size=IMG_SIZE)