import numpy as np
import pandas as pd
from sklearn.linear_model import LinearRegression, ElasticNetCV, SGDRegressor, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingClassifier
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.preprocessing import StandardScaler,  LabelEncoder
import joblib
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from keras import Sequential, layers, regularizers
from keras.layers import Flatten, Dense
from sklearn.ensemble import BaggingRegressor
import pickle
from datetime import datetime, date
import datetime
import matplotlib.pyplot as plt
import keras_cv
import keras_core
import os
import cv2
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense, concatenate


#read in the file right here
#READIN

#normalize the images in the dataset. Might pick one of the two
#should accept a numpy array, use bottom to normlize, should come after the images are resized to what we want. 
#train_images, test_images = train_images / 255.0, test_images / 255.0

#input image size used by Googlenet is 224X224
#9 inception blocks in the archetecture
# four max pooling layers outside the inception blocks
# two between 3-4 blocks and 7-8 blocks 
# dropout layer utilized before linear layer




def makeBlocks(input_layer, f1, f2_conv1, f2_conv3, f3_conv1, f3_conv5, f4): 
    path1 = Conv2D(filters=f1,  kernel_size=(1, 1) ,padding='same', activation='relu')(input_layer)
    
    path2= Conv2D(filters=f2_conv1,  kernel_size=(1, 1) ,padding='same', activation='relu')(input_layer)
    path2= Conv2D(filters=f2_conv3,  kernel_size=(3, 3) ,padding='same', activation='relu')(path2)

    path3 = Conv2D(filters=f3_conv1,  kernel_size=(1, 1) ,padding='same', activation='relu')(input_layer)
    path3 = Conv2D(filters=f3_conv5,  kernel_size=(5, 5) ,padding='same', activation='relu')(path3)

    path4 = MaxPooling2D((3, 3), strides=(1, 1), padding='same')(input_layer)
    path4 = Conv2D(filters=f4,  kernel_size=(1, 1) ,padding='same', activation='relu')(path4)

    outputLayer = concatenate([path1, path2, path3, path4], axis =1)
    
    return outputLayer