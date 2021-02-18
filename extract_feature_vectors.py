#!/usr/bin/env python
# coding: utf-8

# Get all the libraries that're needed 
import pickle
import torch
from torchvision import transforms
from siameseModel import *

import os
import PIL.Image
from tqdm.notebook import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True


# Initialize constant variables
IMAGES_PATH = "/home/vinayak/cleaned_anime_faces/"
FV_PATH = "/home/vinayak/anime_feature_vectors.pkl"
MODEL_PATH = "./enet_model.pth"
IMSIZE = (225, 225)


# Create a model and load the model weights
model = enet_model()
model.load_state_dict(torch.load(MODEL_PATH))
model.eval();


# Write a function to read image and extract feature vector
def predict_feature_vector(img_pth):
    """
    Given an image path, reads the image, converts it into a tensor, normalizes it based on imagenet stats
    and resizes it to (225, 225) standard size; passes it through the model and extracts feature vector
    """
    
    # Convert the image to a tensor and normalize it as per Imagenet stats
    transforms_ = transforms.Compose([transforms.ToTensor(),
                                     transforms.Normalize(mean=[0.485, 0.456, 0.406],
                                                          std=[0.229, 0.224, 0.225]),
                                    ])
    
    # Read the image (function)
    read_img = lambda x: transforms_(PIL.Image.open(x).resize(IMSIZE).convert('RGB')).unsqueeze(0)
    
    # Extract feature vectors
    fv = model(read_img(img_pth))
    
    return fv


# Get a list of all files in our dataset
all_files = []

for root, dirs, files in os.walk(IMAGES_PATH):
    for item in files:
        pth = os.path.join(root, item)
        all_files.append(pth)


# Extract feature vectors and store them in a dictionary
feature_vectors = {}

for file_ in tqdm(all_files, desc = f"Extracting feature vectors from {len(all_files)} images"):
    with torch.no_grad():
        vector = list(predict_feature_vector(file_).numpy()[0])
    feature_vectors[file_] = vector

# Save the feature vectors as a pickle file to load them and analyze them later for nearest neighbours
with open(FV_PATH, "wb") as f:
    pickle.dump(feature_vectors, f)
    f.close()