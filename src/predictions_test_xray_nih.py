'''
    Script for obtaining threshold values and predicstions for a given model and
    individual datasource.
'''

import pandas as pd
from torchvision import transforms
import model_xraytorch_merged as M_xray
import eval_threshold_xray as T_Xray

#suppress pytorch warnings about source code changes
import warnings
warnings.filterwarnings('ignore')

RESULTS_DIRECTORY = "post_analysis_03/"
DATASET_NAME = "nih_03"

# place holder references for wach data set
PATH_TO_IMAGES_NIH = ""
PATH_TO_IMAGES_PAD = ""
PATH_TO_IMAGES_OPENAI = ""

# specific datasource being evaluated
PATH_TO_VAL_FILE_NIH = ""
PATH_TO_TEST_FILE_NIH = ""

PATH_TO_MODEL_NIH = "/checkpoint"
PATH_TO_MODEL_PAD = "/checkpoint"
PATH_TO_MODEL_OPENAI = "/checkpoint"

# model params as established in reproduce-chexnet
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.01
BATCH_SIZE = 16

NUM_WORKERS = 8 # dependant on number of CPUs allocated for work
N_LABELS = 14  # we are predicting 14 labels

if __name__ == '__main__':


    # get data transforms for validation and test data
    standard_transforms = M_xray.get_transforms(split='val')

    categories = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
        'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
        'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

    # pre processing of data/images for dataset
    d_nih_val = M_xray.normalize_dataset_for_provided_pathologies(PATH_TO_IMAGES_NIH, 'nih', desired_pathologies=categories, csv = PATH_TO_VAL_FILE_NIH)
    d_nih_test = M_xray.normalize_dataset_for_provided_pathologies(PATH_TO_IMAGES_NIH, 'nih', desired_pathologies=categories, csv = PATH_TO_TEST_FILE_NIH)

    # translate XRV datasets into compatible reference for CHEXNET repoduction model
    d_nih_val_custom = M_xray.get_model_specific_dataset(d_nih_val, "Image Index", add_png=False, transforms=standard_transforms)
    d_nih_test_custom = M_xray.get_model_specific_dataset(d_nih_test, "Image Index", add_png=False, transforms=standard_transforms)

    print("Creating dataloader dictionary")
    dataloaders = {}
    dataloaders['val'] = M_xray.get_dataloader_for_dataset(d_nih_val_custom, BATCH_SIZE, NUM_WORKERS)
    dataloaders['test'] = M_xray.get_dataloader_for_dataset(d_nih_test_custom, BATCH_SIZE, NUM_WORKERS)

    print("Creating datsetsize dictionary")
    dataset_sizes = {}
    dataset_sizes['val'] = len(d_nih_val)
    dataset_sizes['test'] = len(d_nih_test)

    PRED_LABEL = d_nih_val.pathologies
    print("Starting predictions")

    # obtaining both threshold values and model predictions
    T_Xray.make_pred_multi_label_threshold(PATH_TO_MODEL_NIH, WEIGHT_DECAY, BATCH_SIZE, dataloaders, PRED_LABEL, RESULTS_DIRECTORY, DATASET_NAME, mode='Threshold')
    T_Xray.make_pred_multi_label_threshold(PATH_TO_MODEL_NIH, WEIGHT_DECAY, BATCH_SIZE, dataloaders, PRED_LABEL, RESULTS_DIRECTORY, DATASET_NAME, mode='test')
