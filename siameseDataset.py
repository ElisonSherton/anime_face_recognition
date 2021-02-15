import os
import torch
import random
import PIL.Image
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

IMSIZE = (225, 225)
class siameseDataset(torch.utils.data.Dataset):
    
    def __init__(self, data, dtype = "train"):
        self.images = data
        self.classes = list(set([x.split("/")[-2] for x in self.images]))
        self.dtype = dtype
    
    def __len__(self):
        return len(self.images)
    
    def __getitem__(self, index):
        
        i_pth = self.images[index]
        A, P, N = self.get_triplet(i_pth)
        
        # Create a function to read the image, resize it to a convenient image size
        # convert it to RBG mode (since some have transparency layer alpha) and 
        # finally convert it into a numpy array
        read_img = lambda x: np.asarray(PIL.Image.open(x).resize(IMSIZE).convert('RGB'))
        
        
        A_Img, P_Img, N_Img = map(read_img, (A, P, N))
        return (A_Img, P_Img, N_Img)
    
    def get_triplet(self, i_pth):
        
        # Get the anchor class from the path
        anchor_class = i_pth.split("/")[-2]
        
        # Create a list of all classes, pop the anchor class out from this list and sample 
        # one class which is different than the anchor class
        negative_classes = list(self.classes)
        negative_classes.remove(anchor_class)
        negative_class = random.sample(negative_classes, 1)[0]
        
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