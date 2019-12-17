# MasterMLProject2 : road segmentation

This project was part of a challenge from EPFL course : Machine Learning CS-433 and was hosted on the platform [AIcrowd](https://www.aicrowd.com/).

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

$ pip install sklearn albumentations segmentations_models tta_wrapper

$ easy_install pillow
```

Usage
-----
After installing all the dependencies, got to [AIcrowd](https://www.aicrowd.com/challenges/epfl-ml-road-segmentation-2019/dataset_files)
to download the dataset.

Transfer the *training* and *test_set_images* folders into the same folder as the script.py

Then run:
```
$ python script.py
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
 
