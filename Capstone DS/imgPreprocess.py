import numpy as np
import pandas as pd
import os
import cv2
import numpy as np
from tensorflow import keras

def imgProcess(image_path):
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = np.array(image)
    image = image / 255.0
    return image


