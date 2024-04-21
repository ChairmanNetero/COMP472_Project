import os
from PIL import Image, ImageEnhance
import matplotlib.pyplot as plt

# Source and result folder 
src_folder = '../../Data_Part3/root/surprise'
result_folder = '../../Data_Part3/New/surprise'

# In case result_folder 
os.makedirs(result_folder, exist_ok=True)

# To make it so that it only shows once for the plt images
counter = 0
# Goes through files in the source folder
for filename in os.listdir(src_folder):
    if filename.endswith('.jpg') or filename.endswith('.png'):  # Add or modify if you have different image file types
        # Open the image file
        img = Image.open(os.path.join(src_folder, filename))
        
        # Brighten the image
        enhancer = ImageEnhance.Brightness(img)
        img_bright = enhancer.enhance(1.2)  # increases the image brightness by 20%

        #first image 
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
        img_bright.save(os.path.join(result_folder, filename))

        # Increment the counter so that the plt images only show once
        counter += 1