from PIL import Image

import gc
import os
from tta_wrapper import tta_segmentation
import helpers
import pandas as pd
import numpy as np

N_CYCLES = 6

def predict_model(model_number):
    # loading all the models at each cycles
    models = helpers.load_all_models(
        N_CYCLES,
        model_name = 'efficientnetb7',
        model_number = model_number
    )

    models = [tta_segmentation(mod, h_flip=True, merge='mean') for mod in models]

    ### PREDICTION
    # definition of the DataGenerator of hte images for prediction
    test_generator = helpers.TestGeneratorFolder(
        root_dir = 'test_set_images/test_', 
        image_size = 608
    )

    TEST_SIZE = len(test_generator)
    # directory where the predictions are generated
    prediction_dir = "test_pred{}/".format(model_number)
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

    del models
    gc.collect()

    submission_filename = "pred_model_{}.csv".format(model_number)
    image_filenames = []
    for i in range(1, 51):
        image_filename = "test_pred{}/prediction_{}.png".format(model_number, i)
        image_filenames.append(image_filename)
    helpers.masks_to_submission(submission_filename, *image_filenames, pred = True)
    print("model {} predictions created at -{}-".format(model_number, submission_filename))

if __name__ == '__main':
    predict_model(1)
    predict_model(2)

    preds1 = pd.read_csv('pred_model_1.csv')
    preds2 = pd.read_csv('pred_model_2.csv')

    preds = (preds1['prediction'] + preds2['prediction']) / 2
    del preds1['prediction']

    preds1['prediction'] = np.where(preds > helpers.foreground_threshold, 1, 0)

    file_name = 'result.csv'
    preds1.to_csv(file_name, index = False)
    print("""predictions generated in -{}- file""".format(file_name))    
