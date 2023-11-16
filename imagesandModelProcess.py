import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import googArchitecture
from tensorflow.keras.preprocessing.image import ImageDataGenerator
import matplotlib.pyplot as plt

# Step 1: Define the Data Directory
data_directory = "c:/Users/Jacob Borchert/Desktop/Capstone DS/pictures"

# Step 2: Image Processing (First Code Block)
def imgProcess(image_path):
    # Read and preprocess the image
    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    print("Image Path:", image_path)
    print("Image Shape:", image.shape)
    image = image / 255.0  # Normalize pixel values between 0 and 1
    return image

# Step 2: Data Generator
datagen = ImageDataGenerator(rescale=1./255, validation_split=0.2)

train_generator = datagen.flow_from_directory(
    data_directory,
    target_size=(224, 224),
    batch_size=32, 
    class_mode='sparse',  # or 'categorical' if you have one-hot encoded labels
    subset='training', 
    shuffle=True,
    seed=42
)

validation_generator = datagen.flow_from_directory(
    data_directory,
    target_size=(224, 224),
    batch_size=32,  
    class_mode='sparse',  # or 'categorical' if you have one-hot encoded labels
    subset='validation',  # Use validation split
    shuffle=False,
    seed=42
)

# Step 3: Model Training (Second Code Block)
CNN = googArchitecture.googleNetModelArchitecture()
CNN.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

epochs = 10

history = CNN.fit(train_generator, epochs=epochs, validation_data=validation_generator, verbose=1)

# Plotting the loss over epochs
training_loss = history.history['loss']
validation_loss = history.history['val_loss']

# Plot training & validation loss values
plt.plot(training_loss, label='Training Loss')
plt.plot(validation_loss, label='Validation Loss')
plt.title('Model loss over epochs')
plt.ylabel('Loss')
plt.xlabel('Epoch')
plt.legend()
plt.show()

