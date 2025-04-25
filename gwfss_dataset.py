'''
A script to load the GWFSS dataset.

'''

import os
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable
from PIL import Image
import numpy as np


CLASS_LABELS = [
    {
        "class_id": 0,
        "name": "background",
        "color": [0, 0, 0],
    },
    {
        "class_id": 1,
        "name": "head",
        "color": [132, 255, 50]
    },
    {
        "class_id": 2,
        "name": "stem",
        "color": [255, 132, 50]
    },
    {
        "class_id": 3,
        "name": "leaf",
        "color": [50, 255, 214]
    }
]


class GWFSSDataset(Dataset):
    def __init__(self, root_dir: str, transform: Optional[Callable] = None, split: str = 'train'):
        '''
        This function initializes the GWFSSDataset class.
        dataset_dir is the directory of the GWFSS dataset.
        '''
        
        self.is_train = split == 'train'
        self.root_dir = os.path.join(root_dir, "gwfss_competition_train" if self.is_train else "gwfss_competition_val")
        self.image_dir = os.path.join(self.root_dir, "images")
        if self.is_train:    
            self.mask_dir = os.path.join(self.root_dir, "masks") # directory of mask files
            self.class_id_dir = os.path.join(self.root_dir, "class_id") # directory of class id files
        else:
            self.mask_dir = None # if the dataset is not for training, the mask and class id directories are not needed
            self.class_id_dir = None

        self.transform = transform # transform to apply to the image

        self.image_files = [] # list of image files
        self.mask_files = [] # list of mask files
        self.class_id_files = [] # list of class id files
        self.domain_info = [] # list of domain info (domain1, domain2, etc.)
        self.num_classes = len(CLASS_LABELS) # number of classes (4 in total)

        for file in os.listdir(self.image_dir):
            self.image_files.append(os.path.join(self.image_dir, file))
            self.domain_info.append(file.split('_')[0]) # domain info is the first part of the file name i.e domain1, domain2, etc.
        
        if self.is_train:
            for file in os.listdir(self.mask_dir):
                self.mask_files.append(os.path.join(self.mask_dir, file))
            for file in os.listdir(self.class_id_dir):
                self.class_id_files.append(os.path.join(self.class_id_dir, file))

        self.class_info = {
            'colors': [d['color'] for d in CLASS_LABELS],
            'class_ids': [d['class_id'] for d in CLASS_LABELS], 
            'names': [d['name'] for d in CLASS_LABELS]
        }

    def __normalize__(self, image):
        '''
        This function normalizes the image to be between 0 and 1 (min-max normalization).
        '''
        return image / 255.0 

    def __len__(self):
        return len(self.image_files)

    def __getitem__(self, idx):
        '''
        This function returns the image, class id, and domain for the given index.
        If the image is not a training image, the class_id is set to -1.

        Domain is a string like "domain1" or "domain2" etc.

        For now we dont return the mask. 
        '''
        image_path = self.image_files[idx]
        mask_path = self.mask_files[idx] if self.is_train else None
        class_id_path = self.class_id_files[idx] if self.is_train else None
        domain = self.domain_info[idx]

        image = np.array(Image.open(image_path))
        if self.is_train:
            mask = np.array(Image.open(mask_path))
            class_id_img = np.array(Image.open(class_id_path))
            class_id_img = class_id_img.astype(np.int64)
        else:
            mask = None 
            class_id_img = -1 # if the dataset is not for training, the class id is set to -1

        image = self.__normalize__(image)

        if self.transform is not None:
            image = self.transform(image)

        image = np.transpose(image, (2, 0, 1)).astype(np.float32)
        return image, class_id_img, domain
    



if __name__ == "__main__":
    dataset = GWFSSDataset(root_dir="/Users/vishalned/Desktop/GWFSS/")
    print(len(dataset))
    image, class_id_img, domain = dataset[0]
    print(class_id_img)
    print(image.shape, class_id_img.shape, domain)
