# FinalCapstoneDS
This is a repository designed to run a model that shows if someone is attentive or not using labels and CNN's


Head over to this link and download the zip file: https://github.com/AleDel/deepdreamer-touchdesigner/tree/master/models

From here you want to add in the models folder into this zip file directory

1. Install required Libraires:
import os
import cv2
import numpy as np
import pandas as pd
from sklearn.preprocessing import LabelEncoder, StandardScaler
from sklearn.model_selection import train_test_split, cross_val_score, GridSearchCV, KFold
from sklearn.linear_model import LinearRegression, ElasticNetCV, SGDRegressor, LogisticRegression
from sklearn.tree import DecisionTreeRegressor
from sklearn.ensemble import GradientBoostingRegressor, HistGradientBoostingRegressor, RandomForestRegressor, HistGradientBoostingClassifier, BaggingRegressor
from sklearn.neural_network import MLPRegressor
from sklearn.metrics import mean_squared_error, r2_score, mean_absolute_error
from sklearn.decomposition import PCA
import tensorflow as tf
from tensorflow import keras
from keras import Sequential, layers, regularizers
from keras.layers import Input, Conv2D, MaxPooling2D, AveragePooling2D, Dropout, Flatten, Dense, concatenate, GlobalAveragePooling2D
import joblib, pickle
from datetime import datetime, date
import matplotlib.pyplot as plt


2. download zip with trained model.
3. resize your images that you want to be tested to 224X224 if needed (program should take care of it if you read in a "dataset of images").
4. run the model in terminal.

TRAINED MODEL DATA:
After the model has been run and trained here is the statistics behind the first training set with 10 epochs.
![image](https://github.com/Jborch1/FinalCapstoneDS/assets/122740699/0f7f8fce-ecf8-404b-81d0-ed0a32afe0e0)
