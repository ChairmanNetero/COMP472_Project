import os
import matplotlib.pyplot as plt

# Path to the folder containing class folders
data_folder = '../Data_Part3/New'

# Get the list of class folders
class_folders = [folder for folder in os.listdir(data_folder) if os.path.isdir(os.path.join(data_folder, folder))]

# Count the number of images in each class folder
class_image_counts = {}
for class_folder in class_folders:
    class_images = os.listdir(os.path.join(data_folder, class_folder))
    class_image_counts[class_folder] = len(class_images)

# Plotting the bar graph
plt.figure(figsize=(10, 6))
plt.bar(class_image_counts.keys(), class_image_counts.values(), color='skyblue')
plt.xlabel('Class')
plt.ylabel('Number of Images')
plt.title('Number of Images in Each Class')
plt.xticks(rotation=45, ha='right')
plt.tight_layout()
plt.show()
