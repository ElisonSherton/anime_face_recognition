# Import required libraries
from google_images_download import google_images_download   
from tqdm import tqdm

# Instantiate the class for downloading images
response = google_images_download.googleimagesdownload()   

def download_images(keyword, limit = 1):
    """
    Given the keyword that we're looking for and a limit on max number of images to be downloaded,
    downloads the image and places them in a folder in downloads repo in the current directory structure.
    """
    #creating list of arguments
    arguments = {"keywords": keyword ,
                 "limit": limit , 
                 "print_urls": False,
                 "output_directory": "/home/vinayak/anime_faces"}   

    # Pass the arguments to above function and download images
    paths = response.download(arguments)  

# Read all anime characters from the list of available characters
with open("character_list.txt", "r") as f:
    characters = [x.replace("\n", "") for x in f.readlines()]
    f.close()

# Download 30 images for each character
for person in tqdm(characters, desc = f"Downloading 30 images for {len(characters)} characters."):
    download_images(person, 30)