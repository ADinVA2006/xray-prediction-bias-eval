import pandas as pd
import numpy as np
import random


# you will need to customize PATH_TO_IMAGES to where you have uncompressed images
PATH_FOR_SPLITS = "file_splits/"
PATH_TO_REF_FILE = "reference_files/"
DATASET_NAME = ""

TRAIN_SPLIT = 0.7
VAL_SPLIT = 0.1
TEST_SPLIT = 0.2

print("Starting script")

print("Get numpy array with indices")
original_reference = pd.read_csv(PATH_TO_REF_FILE)

ref_len = len(original_reference)
numpy_arr_of_indices = np.arange(ref_len)

print("Create split values")
n_train = int(TRAIN_SPLIT*ref_len)
n_val = int(VAL_SPLIT*ref_len)
n_test = int(TEST_SPLIT*ref_len)

print('Length of og_ref: {0:d}, training: {1:d}, val:{2:d}, test:{3:d} '.format(ref_len, n_train, n_val, n_test))
print('Sum of splits: {0:d}'.format(n_train + n_val +n_test))

indicies_as_list = numpy_arr_of_indices.tolist()

print('Splitting indices')

indices_for_train = random.sample(indicies_as_list, n_train)
indices_train_removed = np.setdiff1d(indicies_as_list, indices_for_train, assume_unique=False).tolist()

indices_for_val = random.sample(indices_train_removed, n_val)

indices_val_removed = np.setdiff1d(indices_train_removed, indices_for_val, assume_unique=False).tolist()
indices_for_test = random.sample(indices_val_removed, n_test)

print('set diff for last split: ' + str(np.setdiff1d(indices_val_removed, indices_for_test)))

print('Extracting split inidices')

train_split_frame = original_reference.loc[indices_for_train]
val_split_frame = original_reference.loc[indices_for_val]
test_split_frame = original_reference.loc[indices_for_test]

print('Saving reference files')

train_split_frame.to_csv(PATH_FOR_SPLITS + "train_split_" + DATASET_NAME +'.csv', index=False)
val_split_frame.to_csv(PATH_FOR_SPLITS + "val_split_" + DATASET_NAME +'.csv', index=False)
test_split_frame.to_csv(PATH_FOR_SPLITS + "test_split_" + DATASET_NAME +'.csv', index=False)

print("completed splits")
