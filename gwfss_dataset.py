'''
A script to load the GWFSS dataset.

'''

import os
import torch
from torch.utils.data import Dataset
from typing import Optional, Callable
from PIL import Image
import numpy as np
import torchvision.transforms as transforms # for converting the image to a tensor  
import cv2 as cv

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
    
class GWFSSPretrainDataset(Dataset):
    def __init__(self, data_path: str, transform: Optional[Callable] = None):
        '''
        This function initializes the GWFSSPretrainDataset class.
        '''

        self.data_path = os.path.join(data_path, "gwfss_competition_pretrain")
        self.image_files = []
        self.domain_info = []

        # the pretraining data is split into multi domains. Each domain has a folder and within each folder there are images. Since this is a pretraining dataset, we dont have masks or class ids.
        for domain in os.listdir(self.data_path):
            for image in os.listdir(os.path.join(self.data_path, domain)):
                self.image_files.append(os.path.join(self.data_path, domain, image))
                self.domain_info.append(domain)

        self.domain_classes = list(set(self.domain_info))
        self.domain_classes_to_idx = {domain: i for i, domain in enumerate(self.domain_classes)}

        self.transform = transform

    def __len__(self):
        return len(self.image_files)
    
    def edged_img(self, image, minval):
        image_gray = cv.cvtColor(image, cv.COLOR_BGR2GRAY)
        edged_image = cv.Canny(image_gray, minval, 2*minval)
        return edged_image

    def __normalize__(self, image):
        '''
        This function normalizes the image to be between 0 and 1 (min-max normalization).
        '''
        return image / 255.0 

    def __getitem__(self, idx):
        image_path = self.image_files[idx]
        domain = self.domain_info[idx]

        image = np.array(Image.open(image_path))
        edge_image = self.edged_img(image, 100)    

        if np.max(edge_image) > 1:
            edge_image = self.__normalize__(edge_image)

        image = self.__normalize__(image)

        if self.transform is not None:
            image = self.transform(image)
            edge_image = self.transform(edge_image)
        
        image = image.float()
        edge_image = edge_image.float() 

        return image, edge_image, self.domain_classes_to_idx[domain]


if __name__ == "__main__":
    dataset = GWFSSPretrainDataset(
        data_path="/lustre/scratch/WUR/AIN/nedun001/Global-Wheat-Full-Semantic-Segmentation/data",
        transform=transforms.ToTensor()
    )
    print(len(dataset))
    image, domain = dataset[0]
    print(image.shape, domain)
    print(dataset.domain_classes)
