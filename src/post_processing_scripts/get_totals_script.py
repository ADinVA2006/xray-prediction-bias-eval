import pandas as pd
import model_xraytorch_merged as M_xray


# you will need to customize PATH_TO_IMAGES to where you have uncompressed images
# currently just prints totals to screen
PATH_TO_IMAGES_NIH = "<>"
PATH_TO_IMAGES_PAD = "<>"
PATH_TO_IMAGES_OPENAI = "<>"

PATH_TO_TRAIN_FILE_NIH = "<path to split ref>/file_splits/train_split_nih.csv"
PATH_TO_VAL_FILE_NIH = "<path to split ref>/file_splits/val_split_nih.csv"
PATH_TO_TEST_FILE_NIH = "<path to split ref>/file_splits/test_split_nih.csv"
PATH_TO_TRAIN_FILE_PAD = "<path to split ref>/file_splits/train_split_pad_chest.csv"
PATH_TO_VAL_FILE_PAD = "<path to split ref>/file_splits/val_split_pad_chest.csv"
PATH_TO_TEST_FILE_PAD = "<path to split ref>/file_splits/test_split_pad_chest.csv"
PATH_TO_TRAIN_FILE_OPENAI = "<path to split ref>/file_splits/train_split_open_ai.csv"
PATH_TO_VAL_FILE_OPENAI = "/<path to split ref>/file_splits/val_split_open_ai.csv"
PATH_TO_TEST_FILE_OPENAI = "<path to split ref>/file_splits/test_split_open_ai.csv"

WEIGHT_DECAY = 1e-4
LEARNING_RATE = 0.01
BATCH_SIZE = 16
NUM_WORKERS = 8
N_LABELS = 14  # we are predicting 14 labels

print("Starting script")

training_transforms = M_xray.get_transforms(split='train')
standard_transforms = M_xray.get_transforms(split='val')

categories = ['Atelectasis', 'Cardiomegaly', 'Consolidation', 'Edema',
       'Effusion', 'Emphysema', 'Fibrosis', 'Hernia', 'Infiltration', 'Mass',
       'Nodule', 'Pleural_Thickening', 'Pneumonia', 'Pneumothorax']

d_nih_train = M_xray.normalize_dataset_for_provided_pathologies(PATH_TO_IMAGES_NIH, 'nih', desired_pathologies=categories, csv = PATH_TO_TRAIN_FILE_NIH)
d_nih_val = M_xray.normalize_dataset_for_provided_pathologies(PATH_TO_IMAGES_NIH, 'nih', desired_pathologies=categories, csv = PATH_TO_VAL_FILE_NIH)
d_nih_test = M_xray.normalize_dataset_for_provided_pathologies(PATH_TO_IMAGES_NIH, 'nih', desired_pathologies=categories, csv = PATH_TO_TEST_FILE_NIH)
d_open_ai_train = M_xray.normalize_dataset_for_provided_pathologies(PATH_TO_IMAGES_OPENAI, 'open_ai', desired_pathologies=categories, csv = PATH_TO_TRAIN_FILE_OPENAI)
d_open_ai_val = M_xray.normalize_dataset_for_provided_pathologies(PATH_TO_IMAGES_OPENAI, 'open_ai', desired_pathologies=categories, csv = PATH_TO_VAL_FILE_OPENAI)
d_open_ai_test = M_xray.normalize_dataset_for_provided_pathologies(PATH_TO_IMAGES_OPENAI, 'open_ai', desired_pathologies=categories, csv = PATH_TO_TEST_FILE_OPENAI)
d_pad_train = M_xray.normalize_dataset_for_provided_pathologies(PATH_TO_IMAGES_PAD, 'pad', desired_pathologies=categories, csv = PATH_TO_TRAIN_FILE_PAD)
d_pad_val = M_xray.normalize_dataset_for_provided_pathologies(PATH_TO_IMAGES_PAD, 'pad', desired_pathologies=categories, csv = PATH_TO_VAL_FILE_PAD)
d_pad_test = M_xray.normalize_dataset_for_provided_pathologies(PATH_TO_IMAGES_PAD, 'pad', desired_pathologies=categories, csv = PATH_TO_TEST_FILE_PAD)

print('data_set_totals for nih train')
print(d_nih_train.totals)

print('data_set_totals for nih val')
print(d_nih_val.totals)

print('data_set_totals for nih test')
print(d_nih_test.totals)

print('data_set_totals for open_ai train')
print(d_open_ai_train.totals)

print('data_set_totals for open_ai val')
print(d_open_ai_val.totals)

print('data_set_totals for open_ai test')
print(d_open_ai_test.totals)

print('data_set_totals for pad train')
print(d_pad_train.totals)

print('data_set_totals for pad val')
print(d_pad_val.totals)

print('data_set_totals for pad test')
print(d_pad_test.totals)

print("done printing totals")
