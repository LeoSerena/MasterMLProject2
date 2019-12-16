# MasterMLProject2
road segmentation

Installation
------------
To install the dependencies first install anaconda and pip on your computer

Create a new environment with the conda console and activate it:

```
$ conda create *my environment name*

$ conda activate *my environment name*
```

Then run on the conda console:

```
$ conda install sckit-image numpy scipy matplotlib tensorflow keras

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

Summary
-------
