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
import os
import cv2
import keras
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense, concatenate, GlobalAveragePooling2D
from maketheBlocks import makeBlocks
from tensorflow.keras.models import Model



def googleNetModelArchitecture():
    # input layer 
    input_layer = Input(shape = (224, 224, 3))


    X = Conv2D(filters = 64, kernel_size = (7,7), strides = 2, padding = 'valid', activation = 'relu')(input_layer)
    X = MaxPooling2D(pool_size = (3,3), strides = 2)(X)

    X = Conv2D(filters = 64, kernel_size = (1,1), strides = 1, padding = 'same', activation = 'relu')(X)
    X = Conv2D(filters = 192, kernel_size = (3,3), padding = 'same', activation = 'relu')(X)
    X = MaxPooling2D(pool_size= (3,3), strides = 2)(X)

    X = makeBlocks(X, f1 = 64, f2_conv1 = 96, f2_conv3 = 128, f3_conv1 = 16, f3_conv5 = 32, f4 = 32)
    X = makeBlocks(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 192, f3_conv1 = 32, f3_conv5 = 96, f4 = 64)

    X = MaxPooling2D(pool_size= (3,3), strides = 2)(X)
    X = makeBlocks(X, f1 = 192, f2_conv1 = 96, f2_conv3 = 208, f3_conv1 = 16, f3_conv5 = 48, f4 = 64)


    X1 = AveragePooling2D(pool_size = (5,5), strides = 3)(X)
    X1 = Conv2D(filters = 128, kernel_size = (1,1), padding = 'same', activation = 'relu')(X1)
    X1 = Flatten()(X1)
    X1 = Dense(1024, activation = 'relu')(X1)
    X1 = Dropout(0.7)(X1)
    X1 = Dense(5, activation = 'softmax')(X1)

    X = makeBlocks(X, f1 = 160, f2_conv1 = 112, f2_conv3 = 224, f3_conv1 = 24, f3_conv5 = 64, f4 = 64)
    X = makeBlocks(X, f1 = 128, f2_conv1 = 128, f2_conv3 = 256, f3_conv1 = 24, f3_conv5 = 64, f4 = 64)
    X = makeBlocks(X, f1 = 112, f2_conv1 = 144, f2_conv3 = 288, f3_conv1 = 32, f3_conv5 = 64, f4 = 64)

    X2 = AveragePooling2D(pool_size = (5,5), strides = 3)(X)
    X2 = Conv2D(filters = 128, kernel_size = (1,1), padding = 'same', activation = 'relu')(X2)
    X2 = Flatten()(X2)
    X2 = Dense(1024, activation = 'relu')(X2)
    X2 = Dropout(0.7)(X2)
    X2 = Dense(1000, activation = 'softmax')(X2)

    X = makeBlocks(X, f1 = 256, f2_conv1 = 160, f2_conv3 = 320, f3_conv1 = 32, f3_conv5 = 128, f4 = 128)
    X = MaxPooling2D(pool_size = (3,3), strides = 2)(X)
    X = makeBlocks(X, f1 = 256, f2_conv1 = 160, f2_conv3 = 320, f3_conv1 = 32, f3_conv5 = 128, f4 = 128)
    X = makeBlocks(X, f1 = 384, f2_conv1 = 192, f2_conv3 = 384, f3_conv1 = 48, f3_conv5 = 128, f4 = 128)
    X = GlobalAveragePooling2D(name = 'GAPL')(X)
    X = Dropout(0.4)(X)
    CNN = Dense(1000, activation='softmax')(X)
    CNN = Model(inputs=input_layer, outputs=CNN)


    return CNN
