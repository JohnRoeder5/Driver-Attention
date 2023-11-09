import cv2
import numpy as np

# Load pre-trained GoogLeNet/InceptionV1 model
net = cv2.dnn.readNetFromCaffe('models/deploy.prototxt', 'models/bvlc_googlenet.caffemodel')

# Function to classify image using GoogLeNet model
def classify_image(img_path):
    img = cv2.imread(img_path)
    blob = cv2.dnn.blobFromImage(img, 1, (224, 224), (104, 117, 123))
    # Set the input to the network and perform a forward pass to get the predictions
    net.setInput(blob)
    preds = net.forward()

    # Load the class labels
    with open('labels/imagenet-simple-labels.json') as f:
        labels = f.read().strip().split('\n')

    # Get the class label with the highest probability
    idx = np.argmax(preds)
    label = labels[idx]

    # Print the predicted label
    print('Congrats, model found: Predicted Label:', label)

    # Display the image
    cv2.imshow('Image', img)
    cv2.waitKey(0)
    cv2.destroyAllWindows()
# Path to the image you want to classify
#image_path = 'c0/img_208.jpg'
image_path = 'c0/img_344.jpg'

# Call the classify_image function with the image path
classify_image(image_path)




