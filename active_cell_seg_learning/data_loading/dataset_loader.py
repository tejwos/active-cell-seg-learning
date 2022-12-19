import torch
import os
import numpy as np
from skimage.io import imread
from torch.utils import data
from torch.utils.data import DataLoader

import random

import albumentations as A
from albumentations.pytorch import ToTensorV2


def get_loaders(dict_args):
    train_transform, validation_transform = get_transformer(imageHeight=dict_args['img_cols'], 
                                                            imageWidth= dict_args['img_rows'],
                                                            nChannels=  dict_args['n_channels'],
                                                            )
    
    train_data_loader = get_loader(transform=train_transform,
                        path=dict_args['WDB'],
                        mask_suffix=dict_args['mask_suffix'],
                        batch_size=dict_args['batch_size'],
                        number_worker=dict_args['num_workers'],
                        pin_memory=True,
                        shuffel=False, # was true
                        )

    test_data_loader = get_loader(transform=validation_transform,
                        path=dict_args['TDB'],
                        mask_suffix=dict_args['mask_suffix'],
                        batch_size=dict_args['batch_size'],
                        number_worker=dict_args['num_workers'],
                        pin_memory=True,
                        shuffel=False,
                        )

    validation_data_loader = get_loader(transform=validation_transform,
                        path=dict_args['VDB'],
                        mask_suffix=dict_args['mask_suffix'],
                        batch_size=dict_args['batch_size'],
                        number_worker=dict_args['num_workers'],
                        pin_memory=True,
                        shuffel=False,
                        )

    return train_data_loader, test_data_loader, validation_data_loader

def seed_worker(worker_id):
    worker_seed = torch.initial_seed() % 2**32
    np.random.seed(worker_seed)
    random.seed(worker_seed)

def get_g():
    g = torch.Generator()
    g.manual_seed(0)
    return g

def get_loader(transform,
            path: str,
            mask_suffix: str, 
            batch_size: int, 
            number_worker: int ,       
            pin_memory = True,        
            shuffel = True,                
            ):

    data_set = SegmentationDataSetGeneral(
        path=path,
        maskSuffix=mask_suffix,
        transform=transform
    )

    data_loader = DataLoader(
        data_set,
        batch_size=batch_size,
        num_workers=number_worker,
        pin_memory=pin_memory,
        shuffle=shuffel,
        worker_init_fn=seed_worker,     # new
        generator=get_g(),              # new
    )

    return data_loader


def get_transformer(imageHeight:int = 256 , imageWidth:int = 256, nChannels = 2):
    """
    Use Resize, Normalizer and other ...
    For n Channels.
    
    Validation Transform is also a Test Transform.
    """

    train_transform = A.Compose(
            [
                A.Resize(height=imageHeight, width=imageWidth),
                #A.Rotate(limit=35, p=1.0),
                #A.HorizontalFlip(p=0.5),
                #A.VerticalFlip(p=0.1),
                A.Normalize(
                    mean = [0] * nChannels, 
                    std = [1] * nChannels, 
                    max_pixel_value=255.0, 
                ),
                ToTensorV2()
            ],
        )

    # No Flip, Rotate and co in Validation and Test, only Resize
    validation_transform = A.Compose(
        [
            A.Resize(height=imageHeight, width=imageWidth),
            A.Normalize(
                mean=[0] * nChannels, 
                std=[1] * nChannels, 
                max_pixel_value=255.0, 
            ),
            ToTensorV2()
        ],
    )

    return train_transform, validation_transform

class SegmentationDataSetGeneral(data.Dataset):
    def __init__(self,
                 path: str,
                 maskSuffix: str,
                 transform=None
                 ):
        """
        DataSet for given Structure:
            | "Path" #to Main Folder/ Dict
            | |-> "Imgs" #Folder
            | ||-> img_1, img_2, img_3, ... img_n
            | |-> "Msks" #Folder
            | ||-> img_1_mask, img_2_mask, ... img_n_mask

        Imgs: can be with channels size !=3
        Masks: must be 1 channel
        
        Path:       Path to main Folder with 2 Subfolders (Imgs, Msks)
        maskSuffix: The End-Part of Mask-File-Name. Need this for integrity check.
        """         
        self.maskSuffix = maskSuffix
        self.inputPath  = os.path.join(path, "Imgs")
        self.targetPath = os.path.join(path, "Msks")
        self.inputs     = sorted(os.listdir(self.inputPath), key = lambda element: element.split("_")[1])
        self.targets    = sorted(os.listdir(self.targetPath), key = lambda element: element.split("_")[1])
        self.transform  = transform
        self.inputs_dtype = torch.float32
        self.targets_dtype = torch.long

    def __len__(self):
        return len(self.inputs)

    def __getitem__(self, index: int):
        
        ### Test: Belong a given mask to the right image?
        testMsk = os.path.splitext(self.targets[index].replace("_mask", ""))[0] # this one is with "_mask"
        testImg = os.path.splitext(self.inputs[index])[0]
        if testMsk == testImg:
            pass
        else:
            raise ValueError("Name of Image and Mask don't match. Error in DataSet. Check your Data.")
        
        ### Select sample
        inputID    = os.path.join(self.inputPath,  self.inputs[index])
        targetID   = os.path.join(self.targetPath, self.targets[index])

        ### Load input and target
        x =  np.array(imread(inputID))   
        y =  np.array(imread(targetID), dtype=np.float32)

        #print(y.shape, x.shape)
        #print(x.shape)

        ### Preprocessing / Augmentations
        if self.transform is not None:
            augmentations = self.transform(image=x, mask=y)
            x = augmentations["image"]
            y = augmentations["mask"]
        y = y[:,:,0]
        return x, y