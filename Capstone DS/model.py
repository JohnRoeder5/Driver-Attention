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
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense, concatenate, GlobalAveragePooling2D
from maketheBlocks import makeBlocks
from imgPreprocess import imgProcess
from keras.applications import ImageDataGenerator
import googArchitecture


#read in the file right here
#RE

#normalize the images in the dataset. Might pick one of the two
#should accept a numpy array, use bottom to normlize, should come after the images are resized to what we want. 
#train_images, test_images = train_images / 255.0, test_images / 255.0

#input image size used by Googlenet is 224X224
#9 inception blocks in the archetecture
# four max pooling layers outside the inception blocks
# two between 3-4 blocks and 7-8 blocks 
# dropout layer utilized before linear layer




data_directory = 1#the file wit the datast
images = []
labels = []

for class_name in os.listdir(data_directory):
    class_directory = os.path.join(data_directory, class_name)
    if os.path.isdir(class_directory):
        for image_name in os.listdir(class_directory):
            image_path = os.path.join(class_directory, image_name)
            image = imgProcess(image_path)
            images.append(image)
            labels.append(class_name)


images = np.array(images)
labels = np.array(labels)

label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels, test_size= 0.2, random_state=42)




CNN = googArchitecture.googleNetModelArchitecture()
CNN.compile(loss= tf.keras.losses.MeanSquaredError() , optimizer= tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['mae', 'mse'])


#Turn off when doing cross val
epochs = 100 
batchSize = 100
history = CNN.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, verbose=0, validation_split=0.2)  


loss = CNN.evaluate(X_test, y_test)
print("loss: ", loss)
