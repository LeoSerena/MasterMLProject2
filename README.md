# MasterMLProject2 : road segmentation
### TEAM : dabbing_squidwards
### TEAM MEMBERS : 
- Nicolas Brandt 274215 nicolas.brandt-dit-grieurin@epfl.ch
- Leo Marco Serena 263565 leo.serena@epfl.ch
- Rodrigo Granja 241757 rodrigo.soaresgranja@epfl.ch

Description
-----------
This project was part of a challenge from EPFL course : Machine Learning CS-433 and was hosted on the platform [AIcrowd](https://www.aicrowd.com/). The objective was, given satellite images and corresponding masks, to recognize where road are on a set of satellite images.

Content
-------
The folder contains the folowing files:
 - a *script.py* file containing the code implementing 2 different models
 - a *helpers.py* file containing the code of the helper functions for *script.py*
 - a *run.py* calling the scripts of the 2 models, performing the mean of the predictions and generating the final .csv
 - a *repport.pdf* file containing the pdf of the repport 

Installation
------------
To install the dependencies first install [anaconda](https://www.anaconda.com/distribution/) and be sure to have pip on your computer as well as python 3.7.

Create a new environment with the conda console and activate it:

```
$ conda create --name *my environment name*

$ conda activate *my environment name*
```

Then run on the conda console:

```
$ conda install scikit-image numpy scipy matplotlib tensorflow keras pandas

$ pip install sklearn albumentations segmentation_models tta_wrapper pillow
```

Usage
-----
#### Data setup
After installing all the dependencies, got to [AIcrowd](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation-2019/dataset_files)
to download the dataset.

Transfer the *test_set_images* (satellite images for prediction for AIcrowd), *training* (satellite images for training) and *test_set_images* (corresponding masks of the satellite images for training) folders into the same folder as the *script.py*

Then go to the [additionnal dataset link](https://www.kaggle.com/insaff/massachusetts-roads-dataset) and download the training set

Then, create a new folder in the same folder as the *run.py* file and name it *training_augmented*. In the folder, deposit the basic training and masks images in two folders named *images* and *groundtruth* and the 100 first image and masks of the newly downloaded data as well.

Your tree should look like this:

```
.
+-- run.py
+-- script.py
+-- helpers.py
+-- README.md
+-- training
|    |   images
|    |    |    *base satellite images
|    |   groundtruth
|    |    |    *base masks
+-- training_augmented
|    |   images
|    |    |    *base satellite images and additionnal dataset images
|    |   groundtruth
|    |    |    *base masks and additionnal dataset masks
+-- test_set_images
|        *test images*
```

#### Running
All further bash commands should be used in the anaconda console, in the new conda environment *my environment name*:

Now run:
```
$ python run.py
```
This will run the training of the two models as well as merging the predictions, returning a csv named *result.csv*. These are the results of the prediction from the images of the *test_set_images* folder.


#### Memory errors
If memory errors occur, instead of running the two models with one command from the run.py file, please follow the next instructions:

First of all, make sure to have the same configuration up to the point of running ($ python run.py).

Then, perform:
```
$ python script.py 1
```
This will generate a .csv file with the predictions for the first model. Then run
```
$ python script.py 2
```
It will do the same for the second model.

Finally, run
```
$ python run merge
```
This will take the two created predctions and merge them, creating the *result.csv* file

Ressources
----------
Part of the code was inspired by this [github repository](https://github.com/Diyago/ML-DL-scripts/blob/master/DEEP%20LEARNING/segmentation/Segmentation%20pipeline/segmentation%20pipeline.ipynb) by *Инсаф Ашрапов*, especially the DataGenerator and the Albumentations package usage.
For cyclical learning rates, we used the implementation from this [code](https://machinelearningmastery.com/snapshot-ensemble-deep-learning-neural-network/) by *Jason Brownlee*.

Summary
-------
The aim of this project is to detect roads on a dataset of satellite images. The competition metric is the f1 score, and we achieve a public score of 0.922, which is tied for first place as of this writing.
We use a U-net, a segmentation specific neural network (implementation from from the [Segmentation_models](https://github.com/qubvel/segmentation_models) package) and improved our results with various techniques, including: 

 - Selecting the backbone
 - Selecting the loss function
 - Selecting appropriate optimizer and tuning the learning rate
 - Cyclical learning rates and snapshot ensembles
 - Data augmentation
 - Test Time Augmentation
 - External data
 - Model ensembling
 
