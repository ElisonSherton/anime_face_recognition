# The names of images obtained by downloading is very unclean and also there are some images which are
# corrupt/ not recognizable. This repo intends to clean up this mess and make data more organized.

import PIL.Image
import os
import shutil
import glob

# Create a new directory where we would like to transfer all the files, overwrite if already exists
new_pth = "/home/vinayak/random_anime_faces/"
original_images_pth = "/home/vinayak/anime_faces"

if os.path.exists(new_pth):
    shutil.rmtree(new_pth)

os.mkdir(new_pth)

# Check if the file is a .png or .jpg
check_extension = lambda x: x.split(".")[-1] in ["jpg", "png"]
_, char_directory, _ = next(os.walk(original_images_pth))

# Iterate over every character
for character in char_directory:

    new_char_dir = os.path.join(new_pth, character)
    if not os.path.exists(new_char_dir):
        os.mkdir(new_char_dir)

    root, _, char_files = next(os.walk(f"{original_images_pth}/{character}"))

    # Iterate over all files of a particular character
    for idx, file_ in enumerate(char_files, start = 1):

        # Get the name and extension of each image
        fname = file_.split("/")[-1]
        extension = file_.split(".")[-1]

        # Create source and destination paths
        src = os.path.join(root, file_)
        dest = os.path.join(new_char_dir, f"{character.replace(' ', '_')}_{idx}.{extension}")
        
        if check_extension(fname):
            try:
                # Make sure Pillow can read the image
                PIL.Image.open(src)
                shutil.copy(src, dest)
            except Exception as e:
                # If any image is corrupt, print the reason and the name of character for which it is corrupt
                print(str(e), character)
    