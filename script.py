import os
import sys
from PIL import Image
import keras
import tensorflow as tf

from sklearn.utils import shuffle

import pandas as pd

from albumentations import (Blur, Compose, HorizontalFlip, OneOf,
                            RandomContrast, RandomCrop, RandomGamma,
                            ShiftScaleRotate, Transpose, VerticalFlip, Resize)

from keras.callbacks import Callback

from segmentation_models import Unet
from keras.optimizers import SGD
from segmentation_models.losses import jaccard_loss
from segmentation_models.metrics import iou_score,f1_score
from keras import backend
from tta_wrapper import tta_segmentation
import helpers
import gc

tf.set_random_seed(1)


def run(num_model):
    """
    This function trains a model and saves the predictions of the test set in a .csv file

    Parameters
    ----------
    num_model : int
        possible values: 1, 2. The model number.
        First model:
            - batch size of 20
            - crop size of 128
            - 2400 epochs
            - dataset of 100 original images
        Second model:
            - batch size of 2
            - image size of 384 (to work with Unet)
            - 300 epochs
            - dataset of 100 original images plus 100 supplementary images
    """
    ## Setting up DataGenerators for training and validation set
    if num_model == 1:
        BATCH_SIZE = 20
        # crop dimension for data augmentation
        CROP_DIM = 128
        IMG_SIZE = 128
        N_EPOCHS = 2400
        root_dir = 'training/'

    elif num_model == 2:
        BATCH_SIZE = 2
        CROP_DIM = 384
        IMG_SIZE = 384
        N_EPOCHS = 300
        root_dir = 'training_augmented/'

    else:
        print("wrong index")


    # number of cycles in the number of epochs. Thus also is the number of saved
    N_CYCLES = 6

    # Defining the best augmentation
    def aug(crop_dim = CROP_DIM, img_size = IMG_SIZE):
        """
        Parameters
        ----------
        CROP_DIM :int
            Dimensions of the image after being cropped
        IMG_SIZE :int
            Dimensions of the returned image

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
        root_dir = root_dir,
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
    prediction_dir = "test_pred{}/".format(num_model)
    # directory for image to predict
    test_dir = 'test_set_images/test_'

    if not os.path.isdir(prediction_dir):
        os.mkdir(prediction_dir)
    for i in range(1, TEST_SIZE + 1):
        # fetches the test_images
        pimg = helpers.get_prediction_from_ensemble(
            models,
            '{}{}/test_{}.png'.format(test_dir, i, i),
            image_idx = i-1,
            generator = test_generator
        )
        # saves the predictions in ./test_pred*num_model*/prediction_*num_prediction*.png
        Image.fromarray(pimg).save(prediction_dir + "prediction_{}.png".format(i))

    del model
    del models
    gc.collect()


    submission_filename = "pred_model_{}.csv".format(num_model)
    image_filenames = []
    for i in range(1, 51):
        image_filename = "test_pred{}/prediction_{}.png".format(num_model, i)
        image_filenames.append(image_filename)
    helpers.masks_to_submission(submission_filename, *image_filenames, pred = True)
    print("model {} predictions created at -{}-".format(num_model, submission_filename))


if __name__ == '__main__':
    if not len(sys.argv) == 2:  
        print('wrong arguments')
        sys.exit(0)
    num_model = int(sys.argv[1])
    if not (num_model == 1 or num_model == 2):
        print('error, model number must be 1 or 2')
        sys.exit(0)
    run(num_model = num_model)