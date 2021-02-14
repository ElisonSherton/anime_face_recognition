# This script is used to create a train-validation index for the available images
# We shall keep the number of images in the train set to be 80% and those in validation set to be 20%

import os
import random
import pandas as pd

# Set a seed in random module for ensuring reproductibility
random.seed(10)
split_percent = 0.8
data_path = "/home/vinayak/cleaned_anime_faces"

# Create containers to hold paths to train and validation images
train_set = []
validation_set = []

# Iterate over every character folder
for character_folder in os.listdir(data_path):
    # Get all images for a character in a list
    character_images = os.listdir(os.path.join(data_path, character_folder))

    # Get number of images to be kept in train and valid sets
    n_images = len(character_images)
    n_train = int(n_images * split_percent)

    # Sample randomly split_percent proportion of total images 
    # in train and remaining in validation
    train_imgs = random.sample(character_images, n_train)
    valid_imgs = [x for x in character_images if not x in train_imgs]

    # Append the base path to these images
    train_imgs = [os.path.join(data_path, character_folder, x) for x in train_imgs]
    valid_imgs = [os.path.join(data_path, character_folder, x) for x in valid_imgs]

    # Save the sampled images in the container defined outside the loop
    train_set.extend(train_imgs)
    validation_set.extend(valid_imgs)

# Save the indices obtained above into a dataframe and subsequently to a csv file
train_valid_idx = pd.DataFrame({"images": train_set + validation_set,
                                "label": ["train"] * len(train_set) + ["valid"] * len(validation_set)})

train_valid_idx.to_csv("data.csv")