import gzip
import os
import sys
import urllib
import matplotlib.image as mpimg
import matplotlib.pyplot as plt
from PIL import Image
import numpy as np
import time
import random
import keras
import code
import tensorflow.python.platform
import tensorflow as tf

from keras.utils import Sequence
from skimage.io import imread
from sklearn.utils import shuffle

import re
import pandas as pd
import albumentations as albu
from albumentations import Resize

from albumentations import (Blur, Compose, HorizontalFlip, HueSaturationValue,
                            IAAEmboss, IAASharpen, JpegCompression, OneOf,
                            RandomBrightness, RandomBrightnessContrast,
                            RandomContrast, RandomCrop, RandomGamma,
                            RandomRotate90, RGBShift, ShiftScaleRotate,
                            Transpose, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion)

from math import pi
from math import cos
from math import floor
from keras.callbacks import Callback

from segmentation_models import Unet
from keras.optimizers import SGD
from segmentation_models.losses import jaccard_loss,dice_loss
from segmentation_models.metrics import iou_score,f1_score
from keras import backend as K
from keras import backend
from tta_wrapper import tta_segmentation
import re
import helpers

def run_1():
    ## Setting up DataGenerators for training and validation set
    # number of images per gradient step
    BATCH_SIZE = 20
    # crop dimension for data augmentation
    CROP_DIM = 128
    IMG_SIZE = 128
    # number of cycles in the number of epochs. Thus also is the number of saved
    N_CYCLES = 6
    # number of epochs
    N_EPOCHS = 6

    # Defining the best augmentation
    def aug(crop_dim = CROP_DIM, img_size = IMG_SIZE):
        """
        Parameters
        ----------
        CROP_DIM :int
            Dimensions of the image after being cropped
        IMG_SIZE :int
            Dimensions of the image
        Returns
        -------
            The image with composed transformations
        """
        return Compose([
            OneOf([
                RandomContrast(limit = 0.3),
                RandomGamma(),
            ], p=0.5),
            HorizontalFlip(p=0.5),
            VerticalFlip(p=0.5),
            ShiftScaleRotate(p=0.5),
            Blur(p=0.1, blur_limit = 3),
            RandomCrop(width = crop_dim, height = crop_dim, p=1),
            Resize(img_size,img_size,always_apply=True)
        ], p = 1)

    # Defines the DataGenerator of the images for training
    train_generator = helpers.DataGeneratorFolder(
        root_dir = 'training/',
        image_folder = 'images/', 
        mask_folder = 'groundtruth/',
        batch_size = BATCH_SIZE,
        image_size= IMG_SIZE,
        nb_y_features = 1,
        augmentation = aug
    )

    Lr_MAX = 0.1
    # Callback for the cyclic learning rate
    ca = helpers.CosineAnnealingLearningRateSchedule(n_epochs = N_EPOCHS, n_cycles = N_CYCLES, lrate_max = Lr_MAX)
    # List of callbacks
    callbacks = [ca]

    # Definition of the model architecture backbone and loss
    BACKBONE = 'efficientnetb7'
    LOSS = jaccard_loss

    ### MODEL TRAINING AND INSTANCIATION
    ## instantiation of the model
    model = Unet(
        backbone_name = BACKBONE,
        encoder_weights ='imagenet',
        encoder_freeze = False
    )

    # setup of the model using SGD with momentum, jaccard loss and definition of the metrics
    model.compile(
        optimizer = SGD(momentum = 0.9),
        loss = LOSS,
        metrics = [iou_score,f1_score]
    )

    ## trainig of the model
    model.fit_generator(
        train_generator,
        shuffle = True,
        epochs = N_EPOCHS,
        workers = -1,
        use_multiprocessing = True,
        callbacks = callbacks,  
        verbose = 1
    )

    ### PREDICTION
    # definition of the DataGenerator of hte images for prediction
    test_generator = helpers.TestGeneratorFolder(
        root_dir = 'test_set_images/test_', 
        image_size = 608
    )

    # loading all the models at each cycles
    models = helpers.load_all_models(
        N_CYCLES,
        model_name = 'efficientnetb7'
    )

    ## TTA
    # producing the mean of all the models at each cycle
    models = [tta_segmentation(mod, h_flip=True, merge='mean') for mod in models]

    TEST_SIZE = len(test_generator)
    # directory where the predictions are generated
    prediction_dir = "test_pred/"
    # directory for image to predict
    test_dir = 'test_set_images/test_'

    if not os.path.isdir(prediction_dir):
        os.mkdir(prediction_dir)
    for i in range(1, TEST_SIZE + 1):
        pimg = helpers.get_prediction_from_ensemble(
            models,
            '{}{}/test_{}.png'.format(test_dir, i, i),
            image_idx = i-1,
            generator = test_generator
        )
        Image.fromarray(pimg).save(prediction_dir + "prediction_{}.png".format(i))



    if __name__ == '__main__':
        """
        this function is called if we only want the prediction of the first model and creates the final .csv
        """
        run_1()
        submission_filename = "sub_test.csv"
        image_filenames = []
        for i in range(1, 51):
            image_filename = "test_pred/prediction_{}.png".format(i)
            image_filenames.append(image_filename)
        helpers.masks_to_submission(submission_filename, *image_filenames)
    else:
        """
        this is when he file is called from the run.py
        """
        submission_filename = "pred_model_1.csv"
        image_filenames = []
        for i in range(1, 51):
            image_filename = "test_pred/prediction_{}.png".format(i)
            image_filenames.append(image_filename)
        helpers.masks_to_submission(submission_filename, *image_filenames, pred = True)
        print("first model predictions create at -{}-".format(submission_filename))