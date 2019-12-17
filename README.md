# MasterMLProject2 : road segmentation

Description
-----------
This project was part of a challenge from EPFL course : Machine Learning CS-433 and was hosted on the platform [AIcrowd](https://www.aicrowd.com/).

Content
-------
The folder contains the folowing files:
 - 3 script.py files containing the code implementing 3 different models
 - a helpers.py file containing the code of the helper functions for script.py
 - a run.py calling the scripts of the 3 models, performing the mean of the predictions and generating the final .csv

Installation
------------
To install the dependencies first install [anaconda](https://www.anaconda.com/distribution/) and be sure to have pip on your computer

Create a new environment with the conda console and activate it:

```
$ conda create --name *my environment name*

$ conda activate *my environment name*
```

Then run on the conda console:

```
$ conda install scikit-image numpy scipy matplotlib tensorflow keras

$ pip install sklearn albumentations segmentation_models tta_wrapper pillow
```

Usage
-----
After installing all the dependencies, got to [AIcrowd](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation-2019/dataset_files)
to download the dataset.

Transfer the *training* and *test_set_images* folders into the same folder as the script.py

Then go to the [additionnal dataset link](https://www.cs.toronto.edu/~vmnih/data/) and download the training set

Then, create a new folder in the same folder as the run.py file and name it *training_augmented*. In the folder, deposit the basic training and masks images in two folders named *images* and *groundtruth* and put the newly downloaded data as well.

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
```

Then run:
```
$ python run.py
```

Ressources
----------
Part of the code was inspired by this [github repository](https://github.com/Diyago/ML-DL-scripts/blob/master/DEEP%20LEARNING/segmentation/Segmentation%20pipeline/segmentation%20pipeline.ipynb) from *Инсаф Ашрапов*, especially the DataGenerator, the Ablumentations use and the model choice.

Summary
-------
The Objective of this project was to detect Roads given a dataset of satellite images with their groundtruth. 
Since it implies images processing, we used a CNN(Convolutionnal Neural Network) to implement our task, as it is the state-of-the-art method for this kind of problems.

We used the Unet model from the [Segmentation_models](https://github.com/qubvel/segmentation_models) package. and improved it with various techniques, including: 

 - Data augmentation
 - dynamical learning rate
 - model combinations
 - TTA(Test Time Augmentation)
 
