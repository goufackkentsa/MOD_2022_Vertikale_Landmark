import numpy as np
import cv2
import pandas as pd

import pickle
from matplotlib import pyplot as plt
import os

####################################################################
# STEP 1:   READ TRAINING IMAGES AND EXTRACT FEATURES
################################################################
image_dataset = pd.DataFrame()  # Dataframe to capture image features

img_path = 'examples/train/'
for image in os.listdir(img_path):  # iterate through each file
    print(image)

    # Temporary data frame to capture information for each loop.
    df = pd.DataFrame()
    # Reset dataframe to blank after each loop.

    input_img = cv2.imread(img_path + image)  # Read images

    # Check if the input image is RGB or grey and convert to grey if RGB
    if input_img.ndim == 3 and input_img.shape[-1] == 3:
        img = cv2.cvtColor(input_img, cv2.COLOR_BGR2GRAY)
    elif input_img.ndim == 2:
        img = input_img
    else:
        raise Exception("The module works only with grayscale and RGB images!")

################################################################
# START ADDING DATA TO THE DATAFR
