# To use LBP Cascade for face detection
import cv2
import os

src_path = "/home/vinayak/random_anime_faces"
dest_path = "/home/vinayak/half_cropped_images"

# Use the cascade xml file to create a haar cascade detector
cascade = cv2.CascadeClassifier("../resources/lbpcascade_animeface.xml")

def detect(file, output_folder):
    try:
        filename = file.split("/")[-1]
        image = cv2.imread(file, cv2.IMREAD_COLOR)
        gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        
        faces = cascade.detectMultiScale(gray,
                                        # detector options
                                        scaleFactor = 1.1,
                                        minNeighbors = 5,
                                        minSize = (24, 24))

        if len(faces) > 0:
            for internal_idx, (x, y, w, h) in enumerate(faces, start = 1):
                subset = image[x:x + w, y:y + h]
                output_fname, output_ext = ".".join(filename.split(".")[:-1]), filename.split(".")[-1]
                dest_fname = f"{output_folder}/{output_fname}_{str(internal_idx)}.{output_ext}"
                # print(filename, output_fname, dest_fname)
                cv2.imwrite(dest_fname, subset)
        else:
            pass
            # cv2.imwrite(, image)
    except Exception as e:
        print(str(e), file)

if not os.path.exists(dest_path):
    os.mkdir(dest_path)

# Get a list of all the source files
all_files = []
for root, dirs, files in os.walk(src_path):
    for file in files:
        all_files.append(os.path.join(root, file))

# For every source file, apply the face detection lbpcascade and extract 
# the images in destination directory
for file in all_files:

    components = file.split("/")
    character_name, file_name = components[-2], components[-1]
    character_pth = os.path.join(dest_path, character_name)

    if not os.path.exists(character_pth):
        os.mkdir(character_pth)
    
    detect(file, character_pth)
    
