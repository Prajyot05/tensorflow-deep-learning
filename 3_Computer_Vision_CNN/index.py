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