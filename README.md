# COMP472_Project

Github repo: https://github.com/ChairmanNetero/COMP472_Project

## Content
<dl>
  <dt>ExpectationsOfOriginality</dt>
  <dd>Contains the Expectations of Originality files</dd>
  <dt>dataset.txt</dt>
  <dd>Containes provenance of our datasets</dd>
  <dt>SamplesOfDataset</dt>
  <dd>Contains 25 samples of each facial expression</dd>
  <dt>Report.pdf</dt>
  <dd>This is our report</dd>
  <dt>Python</dt>
  <dd>Folder containing our python scripts used</dd>
</dl>

## Python Scripts
### Data Cleaning/Formatting

This command was used to standardize the data

`python format.py`

Note that the inputs and outputs need to be changed to the respective values for each class being formatted. It takes a folder of images and outputs the formatted version of these images.

Also, note that the greyscale line was commented out. This line was only used for the engaged dataset. Since the other datasets are already black and white. So depending on whether this line is commented out or not, you need to make sure that the right image is being outputted.

Lastly, make sure you have the OpenCV library installed

---

This is for re-labeling

`python Re-Labeling_script.py`

This script was to re-label each file properly so that when looking at a picture, the title will tell us what class it's in, and will tell us if it's part of the train or test set.

The script requires 2 parameters to be set at the bottom of the file, the first (path) is the directory where the files you want to rename are, and the second(file_name) is what you want the title of your files 
to be, if you put "test"(as your file_name) and theres 10 files, the output will be, test_1, test_2, test_3, e.t.c ...
Keep in mind, that the script will change all the file names in the directory. 



### Data Visualization
This command was used to plot class distriubtion

`python ClassDistribution.py`

This command was used to plot the 5x5 grid images

`python SampleImages.py`

Install the required libraries using pip:
`pip install numpy matplotlib scikit-learn`


### Part 2 

### Model training 

`Model_Training.py`
This is for model training 
Print in terminal (ordered)
1. Cpu or if conda is being used with gpu
2. All the classes that are recongnized
3. the number of pictures between training, testing and validation
4. Which training model is now training
5. Epoch with train loss, train accuracy, valid loss, valid accuracy
6. Current Valid loss
7. patience counter (for early stopping)
8. print 5-7 until 10 epochs run or it performs early stop
9. print 4-9 until all models are done.

Run with:
`python .\Model_Training.py`

install required libraries using pip3:
`pip3 install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118`

### Performance Evaluation 

`Acrchitectures.py` 
This is used to define the architecture of each model class, its used as a import for eval_images and eval_models

No need to run it since it's just used as an import. 

`eval_images`
`eval_models`
