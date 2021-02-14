import os
import random
import shutil

# Base path where images are stored
base_pth = "/home/vinayak/cleaned_anime_faces"
N = 30

# Loop over every character 
for character in os.listdir(base_pth):
    char_path = os.path.join(base_pth, character)

    # See the number of images per character
    images = os.listdir(char_path)
    deficit = N - len(images)
    
    # If there's fewer than N images in a character, duplicate the character 
    # images till the deficit is succesfully accounted for
    if deficit > 0:
        random_sampled_images = random.sample(images, deficit)

        # From the random sampled images, paste them by appending a suffix 
        # _balancing to hint at the reason of duplication
        for img in random_sampled_images:
            src = os.path.join(char_path, img)
            
            img_name, img_ext = img.split(".")[0], img.split(".")[-1]
            modified_img_name = f"{img_name}_balancing.{img_ext}"
            dest = os.path.join(char_path, modified_img_name)

            shutil.copy(src, dest)