import pandas as pd
import numpy as np
import os
from PIL import Image
from torchvision import transforms

from torch.utils import data
from torchxrayvision.datasets import Dataset

'''
  As the name suggests this is an extension of the XRV merged dataset to work with the reproduce chexnet model.
  The primary difference being the image.convert('RGB') compared to grayscale.
  The XRV datasets were intended to interface with the models included in that library. 
  The XRV library details appropriate use cases and limitations to real-world applications.
'''

class CXRDataset_Modified_for_Xray(Dataset):

    def __init__(self, dataset, image_header_ref, add_png=False, transforms=None):
        super(CXRDataset_Modified_for_Xray, self).__init__()
        self.dataset = dataset
        self.image_header_ref = image_header_ref
        self.add_png = add_png
        self.transforms = transforms
        self.df = dataset.csv
        self.df = self.df.set_index(self.image_header_ref)
        self.PRED_LABEL = self.dataset.pathologies

        RESULT_PATH = "results_merged/"

    def __len__(self):
        return len(self.df)

    def __getitem__(self, idx):

        image_file = self.df.index[idx]
        image_path = self.dataset.imgpath

        if self.add_png is True:
          image_file = image_file + '.png'
        
        image = Image.open(
            os.path.join(image_path, image_file))
        image = image.convert('RGB')

        if self.transforms:
          img = self.transforms(image)
        else:
          img = image

        label = self.dataset.labels[idx].astype(int)
        label[label < 0] = 0
        
        return (img, label, self.df.index[idx])
