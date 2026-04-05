'''
Now we're going to scale up from using 10 classes of the Food101 data to using all of the classes in the Food101 dataset.
'''

# Check if we are using a GPU
# !nvidia-smi

# Get helper functions file
# !wget https://raw.githubusercontent.com/mrdbourke/tensorflow-deep-learning/main/extras/helper_functions.py

# Import all commonly used helper functions
from helper_functions import create_tensorboard_callback, plot_loss_curves, unzip_data, compare_historys, walk_through_dir

# Download data
# !wget https://storage.googleapis.com/ztm_tf_course/food_vision/101_food_classes_10_percent.zip 

unzip_data("101_food_classes_10_percent.zip")

train_dir = "101_food_classes_10_percent/train/"
test_dir = "101_food_classes_10_percent/test/"

# How many images/classes are there?
walk_through_dir("101_food_classes_10_percent")

# Setup data inputs
import tensorflow as tf
IMG_SIZE = (224, 224)
train_data_all_10_percent = tf.keras.preprocessing.image_dataset_from_directory(train_dir,
                                                                                label_mode="categorical",
                                                                                image_size=IMG_SIZE)
                                                                                
test_data = tf.keras.preprocessing.image_dataset_from_directory(test_dir,
                                                                label_mode="categorical",
                                                                image_size=IMG_SIZE,
                                                                shuffle=False) # we're not shuffling because we want to use the order later

'''
Train a model with transfer learning on 10% of 101 food classes
The steps we will follow: 
  1. Create a ModelCheckpoint callback to save our progress during training, this means we could experiment
     with further training later without having to train from scratch every time.

  2. Create a Data augmentation layer right into the model.

  3. Build a headless (no top layers) EfficientNetB0 architecture from tf.keras.applications as our base model.

  4. Build a Dense layer with 101 hidden neurons (same as number of food classes) and softmax activation as the output layer.

  5. Use Categorical crossentropy as the loss function since we're dealing with more than two classes.

  6. Use the Adam optimizer with the default settings.

  7. Fit for 5 full passes on the training data while evaluating on 15% of the test data.
'''

# Create checkpoint callback to save model for later use
checkpoint_path = "101_classes_10_percent_data_model_checkpoint.weights.h5"
checkpoint_callback = tf.keras.callbacks.ModelCheckpoint(checkpoint_path,
                                                         save_weights_only=True, # save only the model weights
                                                         monitor="val_accuracy", # save the model weights which score the best validation accuracy
                                                         save_best_only=True) # only keep the best model weights on file (delete the rest)