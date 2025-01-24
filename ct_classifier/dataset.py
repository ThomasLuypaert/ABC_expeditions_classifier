'''
    PyTorch dataset class for COCO-CT-formatted datasets. Note that you could
    use the official PyTorch MS-COCO wrappers:
    https://pytorch.org/vision/master/generated/torchvision.datasets.CocoDetection.html

    We just hack our way through the COCO JSON files here for demonstration
    purposes.

    See also the MS-COCO format on the official Web page:
    https://cocodataset.org/#format-data

    2022 Benjamin Kellenberger
'''

import os
import re
import torch
from torch.utils.data import Dataset, DataLoader
from torchvision.transforms import Compose, Resize, RandomHorizontalFlip, RandomRotation, RandomCrop, ColorJitter, Normalize, GaussianBlur,Grayscale, RandomApply, ToTensor, Pad
from torchvision import transforms
from PIL import Image
import pandas as pd

class CenterPad:
    def __init__(self, target_size):
        self.target_size = target_size

    def __call__(self, img):
        # Original image dimensions
        width, height = img.size

        # Calculate padding
        pad_width = max((self.target_size[0] - width) // 2, 0)
        pad_height = max((self.target_size[1] - height) // 2, 0)
        padding = (pad_width, pad_height, pad_width, pad_height)

        # Apply padding
        padded_img = transforms.functional.pad(img, padding, fill=0, padding_mode='constant')

        return padded_img

class BlackOut:

    def __call__(self, img):
        return torch.zeros_like(img)
    

class CTDataset(Dataset):

    def __init__(self, cfg, split='train'):
        '''
            Constructor. Here, we collect and index the dataset inputs and
            labels.
        '''
        self.data_root = cfg['data_root']
        self.split = split
        # Transforms. Here's where we could add data augmentation 
        #  For now, we just resize the images to the same dimensions...and convert them to torch.Tensor.
        #  For other transformations see Björn's lecture on August 11 or 

        if cfg["pad"] == False:
            trans_list = [Resize(cfg["image_size"])]

        if cfg["pad"] == True:

            trans_list = [CenterPad((cfg['image_size']))] 

        if split == "train":

            trans_list.append(
                RandomHorizontalFlip(p=cfg['flip_prob']))
            trans_list.append(  
                RandomRotation(degrees = cfg["rot_range"]))       
        
        if cfg["random_crop"] == True and split == "train":
            trans_list.append(RandomApply([RandomCrop(size = (cfg["image_size"][0] * (1 - cfg["crop_perc_red"] / 100)))],p=cfg['crop_prob'])) 

        if cfg["random_brightness"] == True and split == "train":
            trans_list.append(RandomApply([ColorJitter(brightness = cfg["brightness_change"])], p = cfg["brightness_prob"]))

        if cfg["random_contrast"] == True and split == "train":
            trans_list.append(RandomApply([ColorJitter(contrast = cfg["contrast_change"])], p = cfg["contrast_prob"]))

        if cfg["random_blur"] == True and split == "train":
            trans_list.append(RandomApply([GaussianBlur(kernel_size = cfg["blur_kernel"], sigma = (cfg["blur_sig_1"],cfg["blur_sig_2"]))], p = cfg["blur_prob"]))

        if cfg["all_greyscale"] == True and split == "train":
            trans_list.append(Grayscale(num_output_channels=3))
            
        
        trans_list.append(Resize((cfg['image_size'])))
        trans_list.append(ToTensor())

        if cfg["self_destruct"] == True and split == "train":
            trans_list.append(BlackOut())

        if cfg["normalize"] == True:
            trans_list.append(Normalize(mean=[0.485, 0.456, 0.406],std=[0.229, 0.224, 0.225]))

        self.transform = Compose(trans_list)
        
        # index data into list
        self.data = []

        # load metadata file

        annoPath = os.path.join(
            self.data_root, 
            "metadata_small.csv"
        )

        meta = pd.read_csv(annoPath)

        # create custom indices for each category that start at zero. Note: if you have already
        #  had indices for each category, they might not match the new indices.

        categories = meta['category'].unique().tolist()
        #index = categories.index("empty")
        #categories.pop(index)
        #categories.insert(0, "empty")

        self.labels = dict([c, idx]  for idx, c in enumerate(categories))

        self.inv_labels = dict([idx, c]  for idx, c in enumerate(categories))

        # Subsetting metadata based on training or testing data

        if self.split == "train":
            meta = meta[meta["train_test"] == "train"]

        if self.split == "test":
            meta = meta[meta["train_test"] == "validate"]

        if self.split == "none":
            meta = meta

        meta['id'] = meta['file_name'] + '_' + meta['image_name']

        meta['full_path'] = self.data_root + "/" + meta['image_loc']

        # Subsetting the labels dictionary based on training/testing/all data

        self.labels = {k: self.labels[k] for k in meta['category'].unique().tolist()}
        
        # enable filename lookup. Creates image IDs and assigns each ID one filename. 
        #  If your original images have multiple detections per image, this code assumes
        #  that you've saved each detection as one image that is cropped to the size of the
        #  detection, e.g., via megadetector.
        self.images = dict(zip(meta['id'], meta['full_path']))
        
        # since we're doing classification, we're just taking the first annotation per image and drop the rest
        
        for index, row in meta.iterrows():

            self.data.append([row['id'], row['category']])

        for item in self.data:
            item[1] = self.labels.get(item[1])
    

    def __len__(self):
        '''
            Returns the length of the dataset.
        '''
        return len(self.data)

    
    def __getitem__(self, idx):
        '''
            Returns a single data point at given idx.
            Here's where we actually load the image.
        '''
        image_name, label = self.data[idx]              # see line 57 above where we added these two items to the self.data list

        # load image
        image_path = self.images.get(image_name)
        img = Image.open(image_path).convert('RGB')     # the ".convert" makes sure we always get three bands in Red, Green, Blue order

        # transform: see lines 31ff above where we define our transformations
        img_tensor = self.transform(img)

        return img_tensor, label, image_name
    


def create_dataloader(cfg, split='train'):
    '''
        Loads a dataset according to the provided split and wraps it in a
        PyTorch DataLoader object.
    '''
    dataset_instance = CTDataset(cfg, split)        # create an object instance of our CTDataset class
        
    dataLoader = DataLoader(
            dataset=dataset_instance,
            batch_size=cfg['batch_size'],
            shuffle=True,
            num_workers=cfg['num_workers']
        )

    return dataLoader
