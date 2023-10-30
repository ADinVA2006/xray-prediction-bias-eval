import pandas as pd
import numpy as np
import torchxrayvision as xrv
from torchvision import transforms


import os
from PIL import Image
from torchxrayvision.datasets import MergeDataset

'''
  As the name suggests this is an extension of the XRV merged dataset to work with the reproduce chexnet model
  The native XRV MergedDataset did not accommodate the ability to directly split each dataset into train, val, test groups.
  The XRV datasets were intended to interface with the models included in that library. 
  The XRV library details appropriate use cases and limitations to real-world applications.
'''

class CXRDataset_extended_XRV_MergeDataset(MergeDataset):

    def __init__(self, datasets, data_sets_as_df, transforms=None):
        super(CXRDataset_extended_XRV_MergeDataset, self).__init__(datasets)
        self.data_sets_as_df = data_sets_as_df
        self.transforms = transforms
        RESULT_PATH = "results_merged/"

    def __len__(self):
        return len(self.csv)


    def __getitem__(self, idx):

        item = super().__getitem__(idx)

        current_ds_ref = self.which_dataset[idx]
        image_header_ref = self.data_sets_as_df.iloc[current_ds_ref]['imageHeader']
        
        current_dataset = self.datasets[current_ds_ref]
        
        image_file = self.csv[image_header_ref][idx]
        image_path = current_dataset.imgpath

        if self.data_sets_as_df.iloc[current_ds_ref]['add_png'] is True:
          image_file = image_file + '.png'
        
        image = Image.open(
            os.path.join(image_path, image_file))
        image = image.convert('RGB')

        if self.transforms:
          img = self.transforms(image)
        else:
          img = image

        label = item["lab"].astype(int)
        label[label < 0] = 0
        
        return (img, label, self.csv.index[idx])
