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