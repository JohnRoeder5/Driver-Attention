import cv2
import matplotlib.pyplot as plt
import os

# Specify the filename of the image within the "c0" folder
#img_16927.jpg
image_filename = 'img_16927.jpg'  # Replace with the actual image filename

# Construct the full path to the input image
imagePath = os.path.join('c0', image_filename)

# Read the input image
img = cv2.imread(imagePath)

# Check if the image was successfully loaded
if img is not None:
    # Convert the image to grayscale
    gray_image = cv2.cvtColor(img, cv2.COLOR_BGR2GRAY)

    # Load the Haar Cascade classifier for profile face detection
    face_classifier = cv2.CascadeClassifier(cv2.data.haarcascades + "haarcascade_frontalface_default.xml")

    # Perform face detection with adjusted parameters
    face = face_classifier.detectMultiScale(
    gray_image, scaleFactor=1.1, minNeighbors=5, minSize=(40, 40)
    )

    # Draw rectangles around the detected faces
    for (x, y, w, h) in face:
        cv2.rectangle(img, (x, y), (x + w, y + h), (0, 255, 0), 2)

    # Display the image with detected faces
    plt.imshow(cv2.cvtColor(img, cv2.COLOR_BGR2RGB))
    plt.title("Image with Detected Faces")
    plt.axis('off')
    plt.show()

    # Print the number of faces detected
    print(f"Number of faces detected: {len(face)}")

else:
    print("Error: Unable to load the input image.")
