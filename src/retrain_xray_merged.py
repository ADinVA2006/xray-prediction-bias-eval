'''
       This script is intended to serve as an example for the process to retrain a model on a merged dataset leveraging the extended XRV dataset.
       The example below references the datasets evaluated in the paper associated with this repo.
'''

import pandas as pd
import model_xraytorch_merged as M_xray


# you will need to customize PATH_TO_IMAGES to where you have uncompressed images
PATH_TO_IMAGES_NIH = ""
PATH_TO_IMAGES_PAD = ""
PATH_TO_IMAGES_OPENAI = ""

PATH_TO_TRAIN_FILE_NIH = "file_splits/train_split_nih.csv"
PATH_TO_VAL_FILE_NIH = "file_splits/val_split_nih.csv"
PATH_TO_TRAIN_FILE_PAD = "file_splits/train_split_pad_chest.csv"
PATH_TO_VAL_FILE_PAD = "file_splits/val_split_pad_chest.csv"
PATH_TO_TRAIN_FILE_OPENAI = "file_splits/train_split_open_ai.csv"
PATH_TO_VAL_FILE_OPENAI = "file_splits/val_split_open_ai.csv"

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
d_nih_train = M_xray.normalize_dataset_for_provided_pathologies(PATH_TO_IMAGES_NIH, 'nih', desired_pathologies=categories, csv = PATH_TO_TRAIN_FILE_NIH)
d_nih_val = M_xray.normalize_dataset_for_provided_pathologies(PATH_TO_IMAGES_NIH, 'nih', desired_pathologies=categories, csv = PATH_TO_VAL_FILE_NIH)
d_open_ai_train = M_xray.normalize_dataset_for_provided_pathologies(PATH_TO_IMAGES_OPENAI, 'open_ai', desired_pathologies=categories, csv = PATH_TO_TRAIN_FILE_OPENAI)
d_open_ai_val = M_xray.normalize_dataset_for_provided_pathologies(PATH_TO_IMAGES_OPENAI, 'open_ai', desired_pathologies=categories, csv = PATH_TO_VAL_FILE_OPENAI)
d_pad_train = M_xray.normalize_dataset_for_provided_pathologies(PATH_TO_IMAGES_PAD, 'pad', desired_pathologies=categories, csv = PATH_TO_TRAIN_FILE_PAD)
d_pad_val = M_xray.normalize_dataset_for_provided_pathologies(PATH_TO_IMAGES_PAD, 'pad', desired_pathologies=categories, csv = PATH_TO_VAL_FILE_PAD)

# merge datasets into common reference dictionaries for additional processing
normalized_data_sets_train = [
    {'data': d_nih_train, 'imageHeader': 'Image Index', 'add_png': False},
    {'data': d_open_ai_train, 'imageHeader': 'imageid', 'add_png': True},
    {'data': d_pad_train, 'imageHeader': 'ImageID', 'add_png': False}
]

normalized_data_sets_val = [
    {'data': d_nih_val, 'imageHeader': 'Image Index', 'add_png': False},
    {'data': d_open_ai_val, 'imageHeader': 'imageid', 'add_png': True},
    {'data': d_pad_val, 'imageHeader': 'ImageID', 'add_png': False}
]

# extract data components as dataframes to correctly associate model access to XRV interface
datasets_as_df_train = pd.DataFrame(columns = ['data', 'imageHeader', 'add_png'])
datasets_as_df_train = datasets_as_df_train.append(normalized_data_sets_train, ignore_index=True)
dataset_as_list_train = datasets_as_df_train['data'].tolist()

datasets_as_df_val = pd.DataFrame(columns = ['data', 'imageHeader', 'add_png'])
datasets_as_df_val = datasets_as_df_val.append(normalized_data_sets_val, ignore_index=True)
dataset_as_list_val = datasets_as_df_val['data'].tolist()

# obtain a single merged dataset with appropriate tranlations between CHEXNET and XRV
merged_train_custom = M_xray.get_model_specific_merged_dataset(dataset_as_list_train, datasets_as_df_train, transforms=training_transforms)
merged_val_custom = M_xray.get_model_specific_merged_dataset(dataset_as_list_val, datasets_as_df_val, transforms=standard_transforms)

print("Creating dataloader dictionary")
dataloaders = {}
dataloaders['train'] = M_xray.get_dataloader_for_dataset(merged_train_custom, BATCH_SIZE, NUM_WORKERS)
dataloaders['val'] = M_xray.get_dataloader_for_dataset(merged_val_custom, BATCH_SIZE, NUM_WORKERS)

print("Creating datsetsize dictionary")
dataset_sizes = {}
dataset_sizes['train'] = len(merged_train_custom)
dataset_sizes['val'] = len(merged_val_custom)

print("Starting to train model")
trained_model  = M_xray.train_cnn(LEARNING_RATE, WEIGHT_DECAY, N_LABELS, dataloaders, dataset_sizes, 'merged_01')

print("trained_model done training")
