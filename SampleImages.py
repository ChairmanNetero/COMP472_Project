import numpy as np
import random
import matplotlib.pyplot as plt
from sklearn.datasets import load_files

# Path to the folder containing class folders
data_folder = 'Dataset/train'

# Load images using scikit-learn
image_data = load_files(data_folder, shuffle=False)

# Get class names
class_names = [class_folder.split('/')[-1] for class_folder in image_data['target_names']]

# List to store images and their corresponding class labels
image_label_pairs = []

# Iterate over each class
for i, class_name in enumerate(class_names):
    # Get indices of images belonging to the current class
    class_indices = [index for index, target in enumerate(image_data['target']) if target == i]
    # Randomly select 25 images
    random_indices = random.sample(class_indices, min(25, len(class_indices)))

    # Create a new figure
    plt.figure(figsize=(10, 10))

    # Plot images in a 5x5 grid
    images_row = []  # To store images in a row
    for j, index in enumerate(random_indices):
        plt.subplot(5, 5, j + 1)
        image = plt.imread(image_data['filenames'][index])
        plt.imshow(image, cmap='gray')  # Set the colormap to grayscale
        plt.axis('off')
        images_row.append(image)  # Append image to the row
    plt.suptitle('Class: {}'.format(class_name), fontsize=14)
    plt.show()

    # Add the row of images and their labels to the list
    image_label_pairs.append((images_row, [class_name] * len(images_row)))

# Now, image_label_pairs contains pairs of images and their corresponding labels
for images_row, labels in image_label_pairs:
    # Create a new figure for displaying histogram
    plt.figure(figsize=(8, 6))

    # Calculate pixel intensities
    intensities = np.concatenate([image.flatten() for image in images_row])

    # Plot histogram
    plt.hist(intensities, bins=256, color='gray', alpha=0.7)

    # Set title and labels
    plt.title('Pixel Intensity Distribution')
    plt.xlabel('Pixel Intensity')
    plt.ylabel('Frequency')

    # Show the plot
    plt.show()