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