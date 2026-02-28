'''
What is Transfer Learning?
It is taking the patterns (also called weights) another model has learned from another problem and using them for our own problem.

There are two main benefits to using transfer learning:
  1. Can leverage an existing neural network architecture proven to work on problems similar to our own.
  2. Can leverage a working neural network architecture which has already learned patterns on similar data to our own.
     Then we can adapt those patterns to our own data.
'''
import os
os.environ["TF_USE_LEGACY_KERAS"] = "1"

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

'''
Creating models using Tensorflow Hub
Instead of creating our own models layer by layer from scratch,
now majority of our model's layers are going to come from TensorFlow Hub.
'''

# Let's compare two Tensorflow Hub models
resnet_url = "https://tfhub.dev/google/imagenet/resnet_v2_50/feature_vector/4"
efficientnet_url = "https://tfhub.dev/tensorflow/efficientnet/b0/feature-vector/1"

import tensorflow as tf
import tensorflow_hub as hub
from tensorflow.keras import layers

# Function to create a model from a URL
def create_model(model_url, num_classes=10):
  """Takes a TensorFlow Hub URL and creates a Keras Sequential model with it.
  
  Args:
    model_url (str): A TensorFlow Hub feature extraction URL.
    num_classes (int): Number of output neurons in output layer,
      should be equal to number of target classes, default 10.

  Returns:
    An uncompiled Keras Sequential model with model_url as feature
    extractor layer and Dense output layer with num_classes outputs.
  """
  # Download the pretrained model and save it as a Keras layer
  feature_extractor_layer = hub.KerasLayer(model_url,
                                           trainable=False, # freeze the underlying patterns
                                           name='feature_extraction_layer',
                                           input_shape=IMAGE_SHAPE+(3,)) # define the input image shape, (224, 244) + (3,) = (224, 224, 3)
  
  # Create our own model
  model = tf.keras.Sequential([
    feature_extractor_layer, # use the feature extraction layer as the base
    layers.Dense(num_classes, activation='softmax', name='output_layer') # create our own output layer      
  ])

  return model

resnet_model = create_model(resnet_url, num_classes=train_data_10_percent.num_classes)

resnet_model.summary() # The only trainable parameters is the ouput layer

resnet_model.compile(loss="categorical_crossentropy",
                     optimizer=tf.keras.optimizers.Adam(),
                     metrics=["accuracy"])

# Fitting our resnet model to the data
resnet_history = resnet_model.fit(train_data_10_percent,
                                  epochs=5,
                                  steps_per_epoch=len(train_data_10_percent),
                                  validation_data=test_data,
                                  validation_steps=len(test_data),
                                  # Add TensorBoard callback to model (callbacks parameter takes a list)
                                  callbacks=[create_tensorboard_callback(dir_name="tensorflow_hub", # save experiment logs here
                                                                         experiment_name="resnet50V2")]) # name of log files
# The results are incredible (outperforms all models we've built till now, that too with quicker waiting times, and only 10% of our dataset)
# Shows the advantage of using a prebuilt model instead of creating our own from scratch

import pandas as pd
import matplotlib.pyplot as plt

def plot_loss_curves(history):
  """
  Returns separate loss curves for training and validation metrics.
  """ 
  loss = history.history['loss']
  val_loss = history.history['val_loss']

  accuracy = history.history['accuracy']
  val_accuracy = history.history['val_accuracy']

  epochs = range(len(history.history['loss']))

  # Plot loss
  plt.plot(epochs, loss, label='training_loss')
  plt.plot(epochs, val_loss, label='val_loss')
  plt.title('Loss')
  plt.xlabel('Epochs')
  plt.legend()

  # Plot accuracy
  plt.figure()
  plt.plot(epochs, accuracy, label='training_accuracy')
  plt.plot(epochs, val_accuracy, label='val_accuracy')
  plt.title('Accuracy')
  plt.xlabel('Epochs')
  plt.legend()

plot_loss_curves(resnet_history)

# Create EfficientNet model
efficientnet_model = create_model(model_url=efficientnet_url, # use EfficientNetB0 TensorFlow Hub URL
                                  num_classes=train_data_10_percent.num_classes)

# Compile
efficientnet_model.compile(loss='categorical_crossentropy',
                           optimizer=tf.keras.optimizers.Adam(),
                           metrics=['accuracy'])

# Fit
efficientnet_history = efficientnet_model.fit(train_data_10_percent,
                                              epochs=5,
                                              steps_per_epoch=len(train_data_10_percent),
                                              validation_data=test_data,
                                              validation_steps=len(test_data),
                                              callbacks=[create_tensorboard_callback(dir_name="tensorflow_hub", 
                                                                                     # Track logs under different experiment name
                                                                                     experiment_name="efficientnetB0")])

plot_loss_curves(efficientnet_history)

efficientnet_model.summary()

'''
Different types of transfer learning:
  1. "As is" transfer learning - when you take a pretrained model as it is and apply it to your task without any changes.

  2. Feature extraction transfer learning - when you take the underlying patterns (also called weights)
     a pretrained model has learned and adjust its outputs to be more suited to your problem.

  3. Fine-tuning transfer learning is when you take the underlying patterns (also called weights) of a
     pretrained model and adjust (fine-tune) them to your own problem.

A common workflow is to "freeze" all of the learned patterns in the bottom layers of a pretrained model so they're untrainable.
And then train the top 2-3 layers of so the pretrained model can adjust its outputs to your custom data (feature extraction).
After you've trained the top 2-3 layers, you can then gradually "unfreeze" more and more layers and run
the training process on your own data to further fine-tune the pretrained model.

Why train only the top 2-3 layers in feature extraction?
The lower a layer is in a computer vision model as in, the closer it is to the input layer, the larger the features it learn.
For example, a bottom layer in a computer vision model to identify images of cats or dogs might learn the outline of legs, where as,
layers closer to the output might learn the shape of teeth. Often, you'll want the larger features
(learned patterns are also called features) to remain, since these are similar for both animals,
where as, the differences remain in the more fine-grained features.
'''

# See layers in our model
efficientnet_model.layers

# See the amount of layers in the pretrained model
len(efficientnet_model.layers[0].weights)

'''
Comparing our model results using TensorBoard
Note: When uploading things to tensorboard.dev, your experiments are public.
'''

# %load_ext tensorboard
# %tensorboard --logdir tensorflow_hub

# Define a new log directory for these examples to keep them separate from previous model training logs
log_dir_examples = "tensorboard_examples/" + datetime.datetime.now().strftime("%Y%m%d-%H%M%S")
summary_writer = tf.summary.create_file_writer(log_dir_examples)
print(f"TensorBoard logs for these examples will be saved to: {log_dir_examples}")

# To demonstrate scalar logging, let's simulate a simple training loop

# Open the summary writer context to start logging
with summary_writer.as_default():
    # Iterate through a few 'epochs'
    for epoch in range(5):
        # Simulate some loss and accuracy values
        simulated_loss = 1.0 / (epoch + 1)  # Loss decreases over time
        simulated_accuracy = 0.5 + (epoch * 0.1)  # Accuracy increases over time

        # Log the scalar values for 'loss' and 'accuracy' at each 'epoch'
        # tf.summary.scalar() records a single numerical value at a given step (epoch)
        tf.summary.scalar('simulated_loss', simulated_loss, step=epoch)
        tf.summary.scalar('simulated_accuracy', simulated_accuracy, step=epoch)

        print(f"Epoch {epoch}: Loss = {simulated_loss:.4f}, Accuracy = {simulated_accuracy:.4f}")

print("Scalar logging complete.")

# Note: When using `tf.keras.callbacks.TensorBoard` (as in previous cells), 
# scalar logging for loss and metrics happens automatically.

# Let's create a dummy image to log
dummy_image = tf.random.uniform(shape=[1, 64, 64, 3], minval=0, maxval=255, dtype=tf.float32)

# Open the summary writer context
with summary_writer.as_default():
    # Log the image
    # tf.summary.image() records image data. The first dimension is typically batch size, 
    # but for a single image, it's 1. Shape should be [batch_size, height, width, channels].
    tf.summary.image("dummy_input_image", dummy_image, step=0)

    # Let's simulate a processed image (e.g., after some augmentation or feature extraction)
    processed_image = dummy_image * 0.8 + 50 # Simple modification
    tf.summary.image("processed_image_example", processed_image, step=0)

print("Image logging complete.")

# For graph visualization, we need to log a `tf.keras.Model` or `tf.function`.

# Let's define a simple Keras model
def create_simple_model():
    model = models.Sequential([
        layers.Input(shape=(224, 224, 3)), # Define input shape explicitly
        layers.Conv2D(32, (3, 3), activation='relu'),
        layers.MaxPooling2D((2, 2)),
        layers.Flatten(),
        layers.Dense(10, activation='softmax')
    ])
    return model

simple_model = create_simple_model()

# Open the summary writer context
with summary_writer.as_default():
    # Log the Keras model graph
    # To visualize the graph, TensorBoard needs to capture the operations when the model is called.
    # This is typically done by calling the model with some dummy input data.
    tf.summary.trace_on(graph=True, profiler=True) # Start tracing
    _ = simple_model(tf.zeros((1, 224, 224, 3))) # Call the model with dummy input
    tf.summary.trace_export(
        name="simple_model_graph",
        step=0,
        profiler_outdir=log_dir_examples) # Export the trace to the log directory

print("Model graph logging complete.")

# Now, restart TensorBoard to see these new logs. You'll need to run the `tensorboard` magic command again.
# Look for a new experiment directory named something like 'tensorboard_examples/YYYYMMDD-HHMMSS'.