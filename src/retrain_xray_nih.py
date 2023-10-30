'''
       This script is intended to serve as an example for the process to retrain a model on a singular dataset leveraging the extended XRV dataset.
       The example below references the nih dataset.
'''

import pandas as pd
import model_xraytorch_merged as M_xray


# customize PATH_TO_IMAGES to where you have uncompressed images
PATH_TO_IMAGES_NIH = ""
PATH_TO_IMAGES_PAD = ""
PATH_TO_IMAGES_OPENAI = ""

PATH_TO_TRAIN_FILE_NIH = "/file_splits/train_split_nih.csv"
PATH_TO_VAL_FILE_NIH = "/file_splits/val_split_nih.csv"

# model params as established in reproduce-chexnet
WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.01
BATCH_SIZE = 16

NUM_WORKERS = 8 # dependant on number of CPUs allocated for work
N_LABELS = 14  # we are predicting 14 labels

print("Starting script")

# get XRV compatible transforms
training_transforms = M_xray.get_transforms(split='train')
standard_transforms = M_xray.get_transforms(split='val')

categories = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
       'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

# pre processing of data/images for dataset
d_nih_train = M_xray.normalize_dataset_for_provided_pathologies(PATH_TO_IMAGES_NIH, 'nih', desired_pathologies=categories, csv=PATH_TO_TRAIN_FILE_NIH)
d_nih_val = M_xray.normalize_dataset_for_provided_pathologies(PATH_TO_IMAGES_NIH, 'nih', desired_pathologies=categories, csv=PATH_TO_VAL_FILE_NIH)

# translate XRV datasets into compatible reference for CHEXNET repoduction model
d_nih_train_custom = M_xray.get_model_specific_dataset(d_nih_train, "Image Index", add_png=False, transforms=training_transforms)
d_nih_val_custom = M_xray.get_model_specific_dataset(d_nih_val, "Image Index", add_png=False, transforms=standard_transforms)

print("Creating dataloader dictionary")
dataloaders = {}
dataloaders['train'] = M_xray.get_dataloader_for_dataset(d_nih_train_custom, BATCH_SIZE, NUM_WORKERS)
dataloaders['val'] = M_xray.get_dataloader_for_dataset(d_nih_val_custom, BATCH_SIZE, NUM_WORKERS)

print("Creating datsetsize dictionary")
dataset_sizes = {}
dataset_sizes['train'] = len(d_nih_train)
dataset_sizes['val'] = len(d_nih_val)

print("Starting to train model")
trained_model  = M_xray.train_cnn(LEARNING_RATE, WEIGHT_DECAY, N_LABELS, dataloaders, dataset_sizes, 'test_03_nih')

print("trained_model done training")
