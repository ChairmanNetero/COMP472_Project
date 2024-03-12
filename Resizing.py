import cv2
import os

input_dir = 'Dataset/train/neutral'
output_dir = 'output'

if not os.path.exists(output_dir):
    os.makedirs(output_dir)

file_list = os.listdir(input_dir)

for filename in file_list:
    # Read the image
    image = cv2.imread(os.path.join(input_dir, filename))
    
    # Resize the image to 48x48
    resized_image = cv2.resize(image, (192, 192))
    
    # Convert the image to grayscale
    grayscale_image = cv2.cvtColor(resized_image, cv2.COLOR_BGR2GRAY)
    
    # Write the grayscale image to the output directory
    output_filename = os.path.join(output_dir, filename)
    cv2.imwrite(output_filename, grayscale_image)