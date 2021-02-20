#!/usr/bin/env python
# coding: utf-8

# Get all the libraries that're needed 
import pickle
import torch
from torchvision import transforms
from siameseModel import *

import os
import pandas as pd
import PIL.Image
from tqdm import tqdm
from PIL import ImageFile
ImageFile.LOAD_TRUNCATED_IMAGES = True

# Initialize constant variables
IMAGES_PATH = "/home/vinayak/cleaned_anime_faces/"
TRAIN_FV_PATH = "/home/vinayak/anime_feature_vectors_train.pkl"
VALID_FV_PATH = "/home/vinayak/anime_feature_vectors_valid.pkl"
MODEL_PATH = "./resources/enet_model.pth"
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

# Extract feature vectors for train and validation separately and store them in a dictionary
def extract_feature_vectors(dataframe):
    feature_vectors = {}

    for row in tqdm(dataframe.itertuples(), desc = f"Extracting feature vectors from {len(dataframe)} images"):
        file_ = row[1]
        with torch.no_grad():
            vector = list(predict_feature_vector(file_).numpy()[0])
        feature_vectors[file_] = vector
    
    return feature_vectors

# Segregate the dataset in terms of train and validation datasets
df = pd.read_csv("data.csv")
train_df = df[df.label == "train"].reset_index(drop = True)
valid_df = df[df.label == "valid"].reset_index(drop = True)

train_fvs = extract_feature_vectors(train_df)
valid_fvs = extract_feature_vectors(valid_df)

# Save the feature vectors as a pickle file to load them and analyze them later for nearest neighbours
with open(TRAIN_FV_PATH, "wb") as f:
    pickle.dump(train_fvs, f)
    f.close()

with open(VALID_FV_PATH, "wb") as f:
    pickle.dump(valid_fvs, f)
    f.close()