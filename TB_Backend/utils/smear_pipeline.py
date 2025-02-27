"""
This file contains the main pipeline for the smear detection.
The steps that are performed on every tile of the smear are:
    - Preprocessing
    - Thresholding
    - Postprocessing
    - Cropping (optional)
    - Interactive labelling (optional)
    - Dataset creation (optional)
    - Inference (optional)

We are able to count the number of objects in the image and compare it to the
number of objects that are predicted by the model to be bacilli.
"""
import os
import torch
import logging

from utils.generic_utils import *
from utils.cropping import Cropping
from utils.thresholding import Thresholding
from utils.postprocessing import Postprocessing
from utils.n_networks.neural_net import BacilliNet
from utils.inference_visualization import Inference

logger = logging.getLogger(__name__)

def smear_pipeline(smear, loader):
    """This function is the main pipeline for the applying the
    computations on a smear.

    parameters
    ----------
    config: dict
        dictionary with all the parameters for the pipeline
    smear: numpy array
        image of the smear
    loader: class
        class with path to image

    returns
    -------
    total_number_bacilli: int
        total number of bacilli in the smear
    number_of_predicted_bacilli: int
        total number of bacilli predicted by the model
    """
    models = []
    for i in range(1, 6):
        path = os.path.join('utils', 'saved_models', "cnn_results", 'model_' + str(i) + '.pth')
        model_i = BacilliNet()
        model_i.load_state_dict(torch.load(path, map_location=torch.device('cpu'), weights_only=True))
        models.append(model_i)  # get the dataset, with dataset loader

    total_number_bacilli = 0
    number_of_predicted_bacilli = 0
    tiles_bacilli = {}
    logger.info(f"Processing {len(smear)} tiles")
    for i, img in enumerate(smear):


        # Preprocess
        preprocessed_img = rescale(img)

        # Threshold
        threshold = Thresholding(preprocessed_img)
        thresholded_img = threshold.apply()

        # Postprocess
        postprocess = Postprocessing(thresholded_img)
        whole_img_not_cleaned, final_image, num_bacilli, stats = postprocess.apply()
        # clean stats
        stats = clean_stats(stats)
        total_number_bacilli += stats.shape[0]

        # Cropping
        cropped_images = "no images"

        if stats.shape[0] > 1:
            cropping_function = Cropping(img, final_image)
            cropped_images = cropping_function.crop_and_pad()
        else:
            num_bacilli = 0

        if isinstance(cropped_images, str):

            tiles_bacilli[i] = 0

        else:
            # do one of the possible inference
            inference = Inference(cropped_images, stats, final_image)
            red_boxes, green_boxes, predictions = inference.network_prediction(models)
            new_bacilli = green_boxes.shape[0]

            tiles_bacilli[i] = new_bacilli
            number_of_predicted_bacilli += green_boxes.shape[0]

    return number_of_predicted_bacilli, tiles_bacilli, total_number_bacilli
