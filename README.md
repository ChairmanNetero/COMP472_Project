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

`Architectures.py` 
This is used to define the architecture of each model class, its used as a import for eval_images and eval_models

No need to run it since it's just used as an import. 


For the following two scripts, make sure that before running them, you have all three models inside the `python/models` folder and that they are named appropriately (best_model_Main.model, best_model_Variant2.model, best_model_Variant1.model). Also make sure you have the requied packages, which can be seen in the import list at the top of the scripts themselves.

`eval_images.py`
This script is used to evaluate a given image using the main model. It prints out the predicted class for a given image. To run the script, first place yourself in the python folder, then execute `python eval_images.py <path_to_image>`

`eval_models.py`
This script is used to evaluate all three models. It prints out the confusion matrix, precision, recall, f-score and accuracy of each model. To run the script, first place yourself in the python folder, then execute `python eval_models.py`

### Models  
This folder contains all three of the best models saved from `Model_Training`


### Part 3

### Cross Validation Model and scripts

`Model_KCross.py`
This is where it will train with the given dataset, with cross validation of 10 folds, you can define the number of epochs in the fold loop.
it will also print which fold its at and print the macro and micro precision, recall, f1 and accuracy results for each fold's model, at the 
end of the run it will save the model with the best f1 score. As well it calculate the bias metrics on each fold and prints the results. 

run with:
`python Model_KCross.py`

`Brightning_Script.py`
This script simply make the image 20% brighter to mitigate any bias of the data. Must give it a directory folder of images and a result folder directory for after the scripts run 

run with:
`python Brightning_Script.py`

`bias_check.py`
Run metrics on main model saved in the models folder. These metrics include the bias for each subclass.

run with:
`python bias_check.py`

### Models
`best_model_fold_PartII.pt`
This model is the result of training the part II model with the same data using cross validation

`best_model_PartIII_Old.model`
The best model from part III model retraining for better bias after the first attempt to try to fix bias.

`best_model_PartIII.model`
The best model from part III model retraining for better bias after the second and last attempt to fix bias.

### Fold Models 
contains the final models from the final dataset we trained where we saved each fold's model from 0 to 9

### Output texts

`PartII_metric_output_V1.txt`
This is the output of the bias check for the Part II Saved model

`PartIII_metric_output_V2.txt`
This is the output of the final model of Part III for its training fold and all its important data 
