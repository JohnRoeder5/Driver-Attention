import cv2
import pandas as pd
import os
import matplotlib.pyplot as plt
import random

# Read the CSV file
# Replace with the actual path to your CSV file
csv_file_path = "driver_imgs_list.csv"  
image_folder = 'c0'
df = pd.read_csv(csv_file_path)
image_dict = {}

# Choose a random subset size (e.g., 10 images)
subset_size = 10
random_indices = random.sample(range(len(df)), subset_size)

# Iterate through the randomly selected rows of the DataFrame
for index in random_indices:
    row = df.iloc[index]  

    # Get the image filename from the row
    image_filename = row['img']  
    # Construct the full path to the image file
    image_path = os.path.join(image_folder, image_filename)

    # Check if the image file exists
    if os.path.exists(image_path):
        # Load the image using OpenCV
        image = cv2.imread(image_path)

        # Add the image to the dictionary with the index as the key
        image_dict[index] = image

        # Display the image using Matplotlib
        plt.figure()
        plt.imshow(cv2.cvtColor(image, cv2.COLOR_BGR2RGB))
        plt.title(f'Image Index: {index}')
        plt.axis('off')
        plt.show()

        # Print every pixel value in the image
        for row in image:
            for pixel in row:
                print(pixel)

        # Close the figure to free up memory
        plt.close()

# Print the image_dict
for index, image in image_dict.items():
    print(f'Index: {index}, Image Shape: {image.shape}')

