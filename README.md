# COMP472_Project

#Part 1

## Data Cleaning/Formatting

This command was used to standardize the data

`python format.py`

Note that the inputs and outputs need to be changed to the respective values for each class being formatted. It takes a folder of images and outputs the formatted version of these images.

Also, note that the greyscale line was commented out. This line was only used for the engaged dataset. Since the other datasets are already black and white. So depending on whether this line is commented out or not, you need to make sure that the right image is being outputted.


## Data Visualization
This command was used to plot class distriubtion

`python ClassDistribution.py`

This command was used to plot the 5x5 grid images

`python SampleImages.py`

Install the required libraries using pip:
pip install numpy matplotlib scikit-learn

This is for re-labeling
`python Re-Labeling_script.py`
This script was to re-label each file properly so that when looking at a picture, the title will tell us what class it's in, and will tell us if it's part of the train or test set.

The script requires 2 parameters to be set at the bottom of the file, the first (path) is the directory where the files you want to rename are, and the second(file_name) is what you want the title of your files 
to be, if you put "test"(as your file_name) and theres 10 files, the output will be, test_1, test_2, test_3, e.t.c ...
Keep in mind, that the script will change all the file names in the directory. 

