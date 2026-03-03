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