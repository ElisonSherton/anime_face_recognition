import os
import torch
import random
import PIL.Image
import numpy as np
import pandas as pd
from torchvision import transforms
import matplotlib.pyplot as plt

IMSIZE = (225, 225)
class siameseDataset(torch.utils.data.Dataset):
    
    def __init__(self, data, dtype = "train", P = 4, K = 4):
        self.images = data
        
        # In batch hard triplet loss mining, we need to sample P classes and K images per class
        # Save these in the dataset as we will curate the batch using this dataset class only
        self.P = P
        self.K = K
        
        # Store all the classes in a list
        classes = list(set([x.split("/")[-2] for x in self.images]))
        self.classes = classes

        # Create a mapping of class - integer label
        class_mapping = {}
        for idx, item in enumerate(classes):
            class_mapping[item] = idx
        self.class_mapping = class_mapping

        # Save the type of dataset i.e. train/validation/test
        self.dtype = dtype

        # Save the transforms for augmentation in the transforms instance variable
        if self.dtype == "train":
            self.transforms =   transforms.Compose([transforms.Resize(256),
                                                    transforms.RandomCrop(225),
                                                    transforms.RandomHorizontalFlip(),
                                                    transforms.ToTensor(),
                                                    transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                         std=[0.229, 0.224, 0.225]),
                                                ])
        else:
            self.transforms = transforms.Compose([transforms.Resize(225),
                                                  transforms.ToTensor(),
                                                  transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                                       std=[0.229, 0.224, 0.225]),
                                                ])

    def __len__(self):
        return len(self.images)
    
    def get_character_images(self, character_name):
        
        all_character_images = []
        
        # Search in all images which ones have the specified character name
        # and return back a batch of those images only
        for img in self.images:
            if character_name in img:
                all_character_images.append(img)
        
        return all_character_images            
        
    
    def __getitem__(self, index):
        
        # Function to read an image, resize it to specified size and convert to RGB
        read_img = lambda x: self.transforms(PIL.Image.open(x).convert('RGB'))
        
        # Sample P classes randomly
        classes = random.sample(self.classes, self.P)
        
        batch_images = []
        batch_labels = []
        
        # Sample K images from each of the P classes respectively
        for cls in classes:
            char_imgs = random.sample(self.get_character_images(cls), self.K)
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
    
    def get_triplet(self, i_pth):
        
        # Get the anchor class from the path
        anchor_class = i_pth.split("/")[-2]
                
        # Create a pool of images which contain positive class images except
        # the image in question itself and out of them sample one randomly 
        positive_pool = [x for x in self.images if (anchor_class in x) and (x != i_pth)]
        positive_sample = random.sample(positive_pool, 1)[0]
        
        # Create a pool of images which contain negative class images
        # and out of them, sample one randomly
        negative_pool = [x for x in self.images if not anchor_class in x]
        negative_sample = random.sample(negative_pool, 1)[0]
        
        return (i_pth, positive_sample, negative_sample)
    
    def disp_img(self, axis, pth, type_):

        # Read the image, resize it to IMSIZE and convert it to RGB mode
        read_img = lambda x: PIL.Image.open(x).resize(IMSIZE).convert('RGB')

        # Plot the image on an axis object, turn off the markings and set the title
        axis.imshow(read_img(pth))
        axis.axis("off")
        axis.set_title(type_)
    
    def show_sample(self):
        
        # Randomly sample an image out of the entire set of images
        i_pth = random.sample(self.images, 1)[0]
        
        A, P, N = self.get_triplet(i_pth)
        
        fig, ax = plt.subplots(1, 3, figsize = (10, 10))
        
        self.disp_img(ax[0], A, "Anchor")
        self.disp_img(ax[1], P, "Positive")
        self.disp_img(ax[2], N, "Negative")
        
        fig.tight_layout()