import os
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

# Specify the source folder and the destination folder
src_folder = '../../Data_Part3/root/surprise'
dst_folder = '../../Data_Part3/New/surprise'

# Create the destination folder if it doesn't exist
os.makedirs(dst_folder, exist_ok=True)


counter = 0
# Go through all files in the source folder
for filename in os.listdir(src_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Add or modify if you have different image file types
        # Open the image file
        img = Image.open(os.path.join(src_folder, filename))
        
        # Brighten the image
        enhancer = ImageEnhance.Brightness(img)
        img_bright = enhancer.enhance(1.2)  # Increase brightness by 20%

        # If this is the first image, display the original and brightened images
        if counter == 0:
            # Display the original image
            plt.figure(figsize=(10, 5))
            plt.subplot(1, 2, 1)
            plt.title('Before Brightening')
            plt.imshow(img)

            # Display the brightened image
            plt.subplot(1, 2, 2)
            plt.title('After Brightening')
            plt.imshow(img_bright)
            plt.show()

        # Save the brightened image to the destination folder
        img_bright.save(os.path.join(dst_folder, filename))

        # Increment the counter
        counter += 1