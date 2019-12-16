from keras.utils import Sequence
from keras.callbacks import Callback
import numpy as np
import os
from math import pi
from math import cos
from math import floor
from skimage.io import imread
import albumentations as albu
from albumentations import Resize

from albumentations import (Blur, Compose, HorizontalFlip, HueSaturationValue,
                            IAAEmboss, IAASharpen, JpegCompression, OneOf,
                            RandomBrightness, RandomBrightnessContrast,
                            RandomContrast, RandomCrop, RandomGamma,
                            RandomRotate90, RGBShift, ShiftScaleRotate,
                            Transpose, VerticalFlip, ElasticTransform, GridDistortion, OpticalDistortion)

from segmentation_models import Unet
from keras.optimizers import SGD
from segmentation_models.losses import jaccard_loss,dice_loss
from segmentation_models.metrics import iou_score,f1_score
from keras import backend as K
from keras import backend
import matplotlib.image as mpimg

class DataGeneratorFolder(Sequence):
    """
    Class generating samples and corresponding masks

    Parameters
    ----------
    root_dir : String
        path to the directory containing the images and their groundtruth
    image_folder : String
        path to the images dataset from the root
    mask_folder : String
        path to the groundtruth of the images from the root
    batch_size : int
        number of images per grdient step
    image_size : int
        size of the x and y dimensions of the images
    nb_y_features : int
        number of channels of the masks
    augmentation : function
        function applying transforms on the image dataset. If none is given, no transformations will be applied
    shuffle : boolean
        if True, data will be shuffled at every epochs
    """
    def __init__(self,
        root_dir='training/',
        image_folder='images/',
        mask_folder='groundtruth/',
        batch_size=1,
        image_size=400,
        nb_y_features=1, 
        augmentation=None,
        shuffle=True):
        self.image_filenames = sorted([root_dir + image_folder + img for img in os.listdir(root_dir + image_folder)])
        self.mask_names = sorted([root_dir + mask_folder + img for img in os.listdir(root_dir + mask_folder)])
        self.batch_size = batch_size
        self.currentIndex = 0
        self.augmentation = augmentation
        self.image_size = image_size
        self.nb_y_features = nb_y_features
        self.indexes = None
        self.shuffle = shuffle

    def __len__(self):
        return int(np.ceil(len(self.image_filenames) / (self.batch_size)))

    def on_epoch_end(self):
        if self.shuffle:
            self.image_filenames, self.mask_names = shuffle(self.image_filenames, self.mask_names)
        
    def read_image_mask(self, image_name, mask_name):
        return imread(image_name)/255, (imread(mask_name, as_gray=True) > 0).astype(np.int8)

    def __getitem__(self, index):
        # Generate indexes of the batch
        data_index_min = int(index*self.batch_size)
        data_index_max = int(min((index+1)*self.batch_size, len(self.image_filenames)))
        
        indexes = self.image_filenames[data_index_min:data_index_max]

        this_batch_size = len(indexes) # The last batch can be smaller than the others
        
        # Defining dataset
        X = np.empty((this_batch_size, self.image_size, self.image_size, 3), dtype=np.float32)
        y = np.empty((this_batch_size, self.image_size, self.image_size, self.nb_y_features), dtype=np.uint8)
        
        for i, sample_index in enumerate(indexes):

            X_sample, y_sample = self.read_image_mask(self.image_filenames[index * self.batch_size + i], 
                                                    self.mask_names[index * self.batch_size + i])

#             X_sample, y_sample = self.read_image_mask(self.image_filenames[index * 1 + i], 
#                                                       self.mask_names[index * 1 + i])
                 
            # if augmentation is defined, we assume its a train set
            if self.batch_size != 1:
                
                augmented = Resize(height=(X_sample.shape[0]//32)*32,
                                   width=(X_sample.shape[1]//32)*32)(image = X_sample, mask = y_sample)
                X_sample, y_sample = augmented['image'], augmented['mask']
                if self.augmentation is not None:
#                     augmented = self.augmentation(self.image_size)(image=X_sample, mask=y_sample)
                    augmented = self.augmentation()(image=X_sample, mask=y_sample)
                    image_augm = augmented['image']
                    mask_augm = augmented['mask']
                    X[i, ...] = np.clip(image_augm, a_min = 0, a_max=1)
                    y[i, ...] = mask_augm.reshape(mask_augm.shape[0],mask_augm.shape[1],1)
                else:
                    X[i, ...] = np.clip(X_sample, a_min = 0, a_max=1)
                    y[i, ...] = y_sample.reshape(y_sample.shape[0],y_sample.shape[1],1)                    
           
            # if augmentation isnt defined, we assume its a test set. 
            # Because test images can have different sizes we resize it to be divisable by 32
            elif self.augmentation is None and self.batch_size ==1:
                X_sample, y_sample = self.read_image_mask(self.image_filenames[index * 1 + i], 
                                                      self.mask_names[index * 1 + i])
                augmented = Resize(height=(X_sample.shape[0]//32)*32, width=(X_sample.shape[1]//32)*32)(image = X_sample, mask = y_sample)
                X_sample, y_sample = augmented['image'], augmented['mask']

                return X_sample.reshape(1, X_sample.shape[0], X_sample.shape[1], 3).astype(np.float32),\
                       y_sample.reshape(1, X_sample.shape[0], X_sample.shape[1], self.nb_y_features).astype(np.uint8)

        return X, y

class CosineAnnealingLearningRateSchedule(Callback):
    """
    Class of the cyclic learning rate, inherited from callbaks

    Parameters
    ----------
    n_epochs : int
        number of training epochs
    n_cycles : int
        number of cycles
    lrate_max : float
        the maximum learning rate at the start of each cycle
    verbose : int
        if set to 1, generates prints
    """
    def __init__(self, n_epochs, n_cycles, lrate_max, verbose=0):
        self.epochs = n_epochs
        self.cycles = n_cycles
        self.lr_max = lrate_max
        self.lrates = list()

    def cosine_annealing(self, epoch, n_epochs, n_cycles, lrate_max):
        """
        Computes the learning rate evolution for every epoch

        Parameters
        ----------
        epoch : int
            the current epoch
        n_epochs : int
            number of training epochs
        n_cycles : int
            number of cycles
        lrate_max : float
            the maximum learning rate at the start of each cycle

        Returns
        -------
        float
            the updated learning rate
        """
        epochs_per_cycle = floor(n_epochs/n_cycles)
        cos_inner = (pi * (epoch % epochs_per_cycle)) / (epochs_per_cycle)
        return lrate_max/2 * (cos(cos_inner) + 1)

    # calculate and set learning rate at the start of the epoch
    def on_epoch_begin(self, epoch, logs=None):
        """
        Computes the new learning rate and updates the model

        Parameters
        ----------
        epoch : int
            the current epoch
        """
        # calculate learning rate
        lr = self.cosine_annealing(epoch, self.epochs, self.cycles, self.lr_max)
        print(lr)
        # set learning rate
        backend.set_value(self.model.optimizer.lr, lr)
        # log value
        self.lrates.append(lr)
        
    # save models at the end of each cycle
    def on_epoch_end(self, epoch, logs={}):
        """
        Updates the epoch and verifes if the cycle has ended. If so, saves the model parameters

        Parameters
        ----------
        epoch : int
            the current epoch
        """
        # check if we can save model
        epochs_per_cycle = floor(self.epochs / self.cycles)
        if epoch != 0 and (epoch + 1) % epochs_per_cycle == 0:
            # save model to file
            filename = "snapshot_model_%d.h5" % int((epoch + 1) / epochs_per_cycle)
#             self.model.save(filename)
            self.model.save_weights(filename)
#             self.model.save_weights('model_weights.h5')
            print('>saved snapshot %s, epoch %d' % (filename, epoch))

PIXEL_DEPTH = 255

# load models from file
def load_all_models(n_models, model_name = 'efficientnetb7'):
    """
    Loads all parameters of all models and instantiace them

    Parameters
    ----------
    n_models : int
        number of models to load
    model_name : String
        name of the backbone model

    Returns
    -------
    List
        the list of the loaded models
    """
    all_models = list()
    for i in range(n_models):
        filename = 'snapshot_model_' + str(i + 1) + '.h5'
        model = Unet(backbone_name = model_name, encoder_weights='imagenet', encoder_freeze = False)
        model.load_weights(filename)
        all_models.append(model)
        print('>loaded %s' % filename)
    return all_models
    
def img_float_to_uint8(img):
    rimg = img - np.min(img)
    rimg = (rimg / np.max(rimg) * PIXEL_DEPTH).round().astype(np.uint8)
    return rimg  

#return prediction for given model and idx of test generator
def get_prediction(model,idx,generator):
        Xtest = generator.__getitem__(idx)
        predicted = model.predict(np.expand_dims(Xtest[0], axis=0))

        return predicted

#return ensemble prediction for several models
def get_prediction_from_ensemble(models,filename, image_idx,generator):
        preds = None
        for model in models:
            gt_img = get_prediction(model,image_idx,generator).squeeze()
            w = gt_img.shape[0]
            h = gt_img.shape[1]
            gt_img_3c = np.zeros((w, h, 3))
            gt_img8 = gt_img
            gt_img_3c[:, :, 0] = gt_img8
            gt_img_3c[:, :, 1] = gt_img8
            gt_img_3c[:, :, 2] = gt_img8
            if preds is None:
                preds = gt_img_3c
            else:
                preds = preds + gt_img_3c
        cimg = preds/len(models)
        cimg = img_float_to_uint8(cimg)
        return cimg

class TestGeneratorFolder(Sequence):
    """
    DataGenerator class for set to predict

    Parameters
    ----------
    root_dir : String
        path to the dataset
    image_size : size of the dataset images
    """
    def __init__(self,
        root_dir = 'training/', 
        image_size = 608
        ):
        self.image_filenames = ["{}{}/test_{}.png".format(root_dir, i, i) for i in range(1,51)]
        self.currentIndex = 0
        self.image_size = image_size
        
    def __len__(self):
        return len(self.image_filenames)
        
    def read_image(self, image_name):
        return imread(image_name)/255
    
    def __getitem__(self, index):
        X_sample = self.read_image(self.image_filenames[index])
        return X_sample.reshape(1, X_sample.shape[0], X_sample.shape[1], 3).astype(np.float32)

foreground_threshold = 0.25 # percentage of pixels > 1 required to assign a foreground label to a patch

# assign a label to a patch
def patch_to_label(patch):
    df = np.mean(patch)
#     return df
    if df > foreground_threshold:
        return 1
    else:
        return 0


def mask_to_submission_strings(image_filename):
    """Reads a single image and outputs the strings that should go into the submission file"""
    img_number = int(re.search(r"\d+", image_filename).group(0))
    im = mpimg.imread(image_filename)
    patch_size = 16
    for j in range(0, im.shape[1], patch_size):
        for i in range(0, im.shape[0], patch_size):
            patch = im[i:i + patch_size, j:j + patch_size]
            label = patch_to_label(patch)
            yield("{:03d}_{}_{},{}".format(img_number, j, i, label))


def masks_to_submission(submission_filename, *image_filenames):
    """Converts images into a submission file"""
    with open(submission_filename, 'w') as f:
        f.write('id,prediction\n')
        for fn in image_filenames[0:]:
            f.writelines('{}\n'.format(s) for s in mask_to_submission_strings(fn))