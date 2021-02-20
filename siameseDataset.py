import os
import torch
import random
import PIL.Image
import numpy as np
import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt

from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

IMSIZE = (225, 225)

class siameseDataset(torch.utils.data.Dataset):
    
    def __init__(self, data, dtype = "train", P = 4, K = 4):
        self.images = data
        self.images_selected = {k:False for k in self.images}
        
        
        # In batch hard triplet loss mining, we need to sample P classes and K images per class
        # Save these in the dataset as we will curate the batch using this dataset class only
        if dtype == "train":
            self.limit = 6
        else:
            self.limit = 2
        self.P = P
        self.K = K
        
        # Store all the classes in a list
        classes = list(set([x.split("/")[-2] for x in self.images]))
        self.classes = classes
        self.characters_selected = {k:0 for k in self.classes}

        # Create a mapping of class - integer label
        class_mapping = {}
        for idx, item in enumerate(classes):
            class_mapping[item] = idx
        self.class_mapping = class_mapping

        # Save the type of dataset i.e. train/validation/test
        self.dtype = dtype

        # Save the transforms for augmentation in the transforms instance variable
        if self.dtype == "train":
            self.transforms =   transforms.Compose([transforms.RandomRotation(degrees = 10),
                                                    transforms.RandomGrayscale(),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225]),
                                                ])
        else:
            self.transforms = transforms.Compose([transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225]),
                                                ])

    def __len__(self):
        return int(len(self.images) // (self.P * self.K))
    
    def sample_characters(self):
        
        all_classes = list(self.classes)
        random.shuffle(all_classes)
        
        char_list = []
        
        while len(char_list) < self.P:
            
            char_ = random.sample(all_classes, 1)[0]
            
            if (self.characters_selected[char_] < self.limit):
                char_list.append(char_)
                self.characters_selected[char_] += 1
                
        return char_list

    
    def sample_character_images(self, character_name):
        
        all_character_images = []
        
        # Search in all images which ones have the specified character name
        # and return back a batch of those images only
        for img in self.images:
            
            if len(all_character_images) == self.K:
                break
                
            if (character_name in img) and (not self.images_selected[img]):
                all_character_images.append(img)
                self.images_selected[img] = True
        
        return all_character_images            
        
    
    def __getitem__(self, index):
        
        # Function to read an image, resize it to specified size and convert to RGB
        read_img = lambda x: self.transforms(PIL.Image.open(x).resize(IMSIZE).convert('RGB'))
        
        # Sample P classes randomly
        classes = self.sample_characters()
        
        batch_images = []
        batch_labels = []
        
        # Sample K images from each of the P classes respectively
        for cls in classes:
            char_imgs = self.sample_character_images(cls)
            batch_images.extend(char_imgs)
            batch_labels.extend([self.class_mapping[cls]] * self.K)
        
        # Create a array of all images by reading them using the lambda function defined above
        images = []
        for img in batch_images:
            images.append(read_img(img).unsqueeze(0))
        images = torch.cat(images, dim = 0)

        # Create a numpy array of labels out of the list created above
        batch_labels = np.array(batch_labels)
        
        return (images, batch_labels)