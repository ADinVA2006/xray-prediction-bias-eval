'''
    Utility class for common methods to support training and or retraining models on various data splits.
    Adapted from reproduce-chexnet model.py to accomodate extended dataset.
'''

from __future__ import print_function, division

# pytorch imports
import torch
import torch.nn as nn
import torch.optim as optim
from torch.autograd import Variable
import torchvision
import torchxrayvision as xrv
from torchvision import models, transforms
from torchvision import transforms

# image imports
from PIL import Image

# general imports
import os
import time
from shutil import rmtree

# data science imports
import pandas as pd
import numpy as np
import csv

import cxr_dataset_for_x_ray as CXR
import cxr_dataset_extending_xray as CXR_MERGED
import eval_model as E


use_gpu = torch.cuda.is_available()
gpu_count = torch.cuda.device_count()
print("Available GPU count:" + str(gpu_count))

def checkpoint(model, best_loss, epoch, LR, model_identifier):
    """
    Saves checkpoint of torchvision model during training.

    Args:
        model: torchvision model to be saved
        best_loss: best val loss achieved so far in training
        epoch: current epoch of training
        LR: current learning rate in training
    Returns:
        None
    """

    print('saving')
    state = {
        'model': model,
        'best_loss': best_loss,
        'epoch': epoch,
        'rng_state': torch.get_rng_state(),
        'LR': LR
    }

    torch.save(state, 'results_' + model_identifier + '/checkpoint')

def train_model(
        model,
        criterion,
        optimizer,
        LR,
        num_epochs,
        dataloaders,
        dataset_sizes,
        weight_decay,
        model_identifier):
    """
    Fine tunes torchvision model to provided data.

    Args:
        model: torchvision model to be finetuned (densenet-121 in this case)
        criterion: loss criterion (binary cross entropy loss, BCELoss)
        optimizer: optimizer to use in training (SGD)
        LR: learning rate
        num_epochs: continue training up to this many epochs
        dataloaders: pytorch train and val dataloaders
        dataset_sizes: length of train and val datasets
        weight_decay: weight decay parameter we use in SGD with momentum
    Returns:
        model: trained torchvision model
        best_epoch: epoch on which best model val loss was obtained

    """
    since = time.time()

    start_epoch = 1
    best_loss = 999999
    best_epoch = -1
    last_train_loss = -1

    # iterate over epochs
    for epoch in range(start_epoch, num_epochs + 1):
        print('Epoch {}/{}'.format(epoch, num_epochs))
        print('-' * 10)

        # set model to train or eval mode based on whether we are in train or
        # val; necessary to get correct predictions given batchnorm
        for phase in ['train', 'val']:
            if phase == 'train':
                model.train(True)
            else:
                model.train(False)

            running_loss = 0.0

            i = 0
            total_done = 0
            # iterate over all data in train/val dataloader:
            for data in dataloaders[phase]:
                i += 1
                inputs, labels, _ = data
                batch_size = inputs.shape[0]
                inputs = Variable(inputs.cuda())
                labels = Variable(labels.cuda()).float()
                outputs = model(inputs)

                # calculate gradient and update parameters in train phase
                optimizer.zero_grad()
                loss = criterion(outputs, labels)
                if phase == 'train':
                    loss.backward()
                    optimizer.step()

                running_loss += loss.data * batch_size

            epoch_loss = running_loss / dataset_sizes[phase]

            if phase == 'train':
                last_train_loss = epoch_loss

            print(phase + ' epoch {}:loss {:.4f} with data size {}'.format(
                epoch, epoch_loss, dataset_sizes[phase]))

            # decay learning rate if no val loss improvement in this epoch

            if phase == 'val' and epoch_loss > best_loss:
                print("decay loss from " + str(LR) + " to " +
                      str(LR / 10) + " as not seeing improvement in val loss")
                LR = LR / 10
                # create new optimizer with lower learning rate
                optimizer = optim.SGD(
                    filter(
                        lambda p: p.requires_grad,
                        model.parameters()),
                    lr=LR,
                    momentum=0.9,
                    weight_decay=weight_decay)
                print("created new optimizer with LR " + str(LR))

            # checkpoint model if has best val loss yet
            if phase == 'val' and epoch_loss < best_loss:
                best_loss = epoch_loss
                best_epoch = epoch
                checkpoint(model, best_loss, epoch, LR, model_identifier)

            # log training and validation loss over each epoch
            if phase == 'val':
                with open('results_' + model_identifier  + '/log_train_' + model_identifier, 'a') as logfile:
                    logwriter = csv.writer(logfile, delimiter=',')
                    if(epoch == 1):
                        logwriter.writerow(["epoch", "train_loss", "val_loss"])
                    logwriter.writerow([epoch, last_train_loss, epoch_loss])

        total_done += batch_size
        if(total_done % (100 * batch_size) == 0):
            print("completed " + str(total_done) + " so far in epoch")

        # break if no val loss improvement in 3 epochs
        if ((epoch - best_epoch) >= 3):
            print("no improvement in 3 epochs, break")
            break

    time_elapsed = time.time() - since
    print('Training complete in {:.0f}m {:.0f}s'.format(
        time_elapsed // 60, time_elapsed % 60))

    # load best model weights to return
    checkpoint_best = torch.load('results_' + model_identifier + '/checkpoint')
    model = checkpoint_best['model']

    return model, best_epoch

def train_cnn(LR, WEIGHT_DECAY, N_LABELS, dataloaders, dataset_sizes, model_identifier):
    """
    Train torchvision model to NIH data given high level hyperparameters.

    Args:
        LR: learning rate
        WEIGHT_DECAY: weight decay parameter for SGD
        N_LABELS: number of labels to be predicted
        dataloaders: pytorch train and val dataloaders
        dataset_sizes: length of train and val datasets

    Returns:
        model: resutling from training
    """
    NUM_EPOCHS = 100

    try:
        rmtree('results/')
    except BaseException:
        pass  # directory doesn't yet exist, no need to clear it
    os.makedirs("results/")


    # please do not attempt to train without GPU as will take excessively long
    if not use_gpu:
        raise ValueError("Error, requires GPU")
    model = models.densenet121(pretrained=True)
    num_ftrs = model.classifier.in_features
    # add final layer with # outputs in same dimension of labels with sigmoid
    # activation
    model.classifier = nn.Sequential(
        nn.Linear(num_ftrs, N_LABELS), nn.Sigmoid())

    # put model on GPU
    model = model.cuda()

    # define criterion, optimizer for training
    criterion = nn.BCELoss()
    optimizer = optim.SGD(
        filter(
            lambda p: p.requires_grad,
            model.parameters()),
        lr=LR,
        momentum=0.9,
        weight_decay=WEIGHT_DECAY)

    # train model
    model, best_epoch = train_model(model, criterion, optimizer, LR, num_epochs=NUM_EPOCHS,
                                    dataloaders=dataloaders, dataset_sizes=dataset_sizes, weight_decay=WEIGHT_DECAY, model_identifier=model_identifier)

    print("training completed")

    return model

def get_transforms(split=None):
    '''
        Method to obtain the image transforms for the various stages of the model cycle.
        Flips are only used in the training stages.
        Args:
            split: stage (train, val, Threshold, test) in pipeline
        Returns:
            The appropriate transforms according the stage of development.
    '''
    
    # use imagenet mean,std for normalization
    mean = [0.485, 0.456, 0.406]
    std = [0.229, 0.224, 0.225]

    if split == 'train':
    # define torchvision transforms
        data_transform =  transforms.Compose([
            transforms.RandomHorizontalFlip(),
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])
    else:
        data_transform =  transforms.Compose([
            transforms.Resize(224),
            transforms.CenterCrop(224),
            transforms.ToTensor(),
            transforms.Normalize(mean, std)
        ])

    return data_transform

def normalize_dataset_for_provided_pathologies(PATH_TO_IMAGES, dataset_code, desired_pathologies, csv=None):
    '''
        This method normalizes the fields of the sources data to the XRV Dataset fields.
        Args:
            PATH_TO_IMAGES: path to datasource images
            dataset_code: currently supports: nih, pad, open_ai
            data_transform: transforms to be applied to data images
            desired_classes: classes being evaluated by the model
            csv: path to file for split, default=None            
        Returns:
            XRV dataset corresponding to source information.
    '''
    xrv_dataset = None
    if dataset_code == 'nih':
        xrv_dataset = xrv.datasets.NIH_Dataset(imgpath=PATH_TO_IMAGES, csvpath=csv)
    elif dataset_code == 'pad':
        xrv_dataset = xrv.datasets.PC_Dataset(imgpath=PATH_TO_IMAGES, csvpath=csv)

    elif dataset_code =='open_ai':
        xrv_dataset = xrv.datasets.Openi_Dataset(imgpath=PATH_TO_IMAGES, dicomcsv_path=csv)
    else:
        raise NotImplementedError("Dataset code provided is not currently supported.")

    xrv.datasets.relabel_dataset(desired_pathologies, xrv_dataset)

    return xrv_dataset

def get_model_specific_merged_dataset(merged_data, data_sets_as_df, transforms=None):
    '''
        Args:
            merged_data: XRV individual datasets merged in one data objeect
            data_sets_as_df: pandas dataframe of merged data that aligns required column values for ea merged data set
            transforms: set of transforms to be applied to merged images
        Returns:
            merged dataset extended from XRV data for interoperability with ChexNet model.
    '''
    return CXR_MERGED.CXRDataset_extended_XRV_MergeDataset(merged_data, data_sets_as_df, transforms)

def get_model_specific_dataset(Dataset, image_header_ref, add_png=False, transforms=None):
    '''
        Intended for use on a single data source
        Args:
            Dataset: XRV dataset after initial processing
            image_header_ref: As it sounds, the column name for the image reference
            add_png: XRV reference file requires additionf of .`png` file extension to read image
            transforms: set of transforms to be applied to merged images
        Returns:
            merged dataset extended from XRV data for interoperability with ChexNet model.
    '''
    return CXR.CXRDataset_Modified_for_Xray(Dataset, image_header_ref, add_png, transforms)

def get_dataloader_for_dataset(dataset, BATCH_SIZE, NUM_WORKERS):
    '''
        Args:
            dataset: compatible with both CHX model and XRV references
            BATCH_SIZE: Batch processing chunk size
            NUM_WORKERS: Likely should be equiv to the number of CPUs
        Returns:
            Creates dataloader with parameter and values from reference script.
    '''

    dataloader = torch.utils.data.DataLoader(
        dataset,
        batch_size=BATCH_SIZE,
        shuffle=True,
        num_workers=NUM_WORKERS)

    return dataloader

