import os
import cv2
import numpy as np
from sklearn.preprocessing import LabelEncoder
from sklearn.model_selection import train_test_split
import tensorflow as tf
import googArchitecture
import matplotlib as plt 
# Import your googArchitecture module containing the model architecture
# net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt', 'models/bvlc_googlenet.caffemodel')

# Step 1: Define the Data Directory
data_directory = "c:/Users/Jacob Borchert/Desktop/Capstone DS/pictures"


# Step 2: Image Processing (First Code Block)
def imgProcess(image_path):
    # Read and preprocess the image
    image = cv2.imread(image_path)
    print("Image Path:", image_path)
    print("Image Shape:", image.shape)
    # Rest of the processing code

    image = cv2.imread(image_path)
    image = cv2.resize(image, (224, 224))
    image = image / 255.0  # Normalize pixel values between 0 and 1
    return image

images = []
labels = []

for class_name in os.listdir(data_directory):
    class_directory = os.path.join(data_directory, class_name)
    if os.path.isdir(class_directory):
        class_images = []
        for image_name in os.listdir(class_directory):
            image_path = os.path.join(class_directory, image_name)
            image = imgProcess(image_path)  
            if image is not None:
                class_images.append(image)
            else:
                print(f"Skipping image: {image_path}")
        print(f"Class {class_name} has {len(class_images)} samples.")
        images.extend(class_images)
        labels.extend([class_name] * len(class_images))

images = np.array(images)
labels = np.array(labels)


# Step 3: Model Training (Second Code Block)
label_encoder = LabelEncoder()
labels_encoded = label_encoder.fit_transform(labels)

X_train, X_test, y_train, y_test = train_test_split(images, labels_encoded, test_size=0.2, random_state=42)

CNN = googArchitecture.googleNetModelArchitecture()
CNN.compile(loss=tf.keras.losses.SparseCategoricalCrossentropy(), optimizer=tf.keras.optimizers.Adam(learning_rate=0.001), metrics=['accuracy'])

epochs = 10
batchSize = 100
history = CNN.fit(X_train, y_train, epochs=epochs, batch_size=batchSize, verbose=1, validation_split=0.2)

loss = CNN.evaluate(X_test, y_test)
print("Loss and Accuracy: ", loss)

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
