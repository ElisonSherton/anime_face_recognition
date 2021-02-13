# To use HAAR Cascade for face detection
import cv2
import os

# Use the cascade xml file to create a haar cascade detector
cascade = cv2.CascadeClassifier("./lbpcascade_animeface.xml")

def detect(filename, idx = 1):
    try:
        image = cv2.imread(filename, cv2.IMREAD_COLOR)
        gray = cv2.equalizeHist(cv2.cvtColor(image, cv2.COLOR_BGR2GRAY))
        
        faces = cascade.detectMultiScale(gray,
                                        # detector options
                                        scaleFactor = 1.1,
                                        minNeighbors = 5,
                                        minSize = (24, 24))

        if len(faces) > 0:
            for internal_idx, (x, y, w, h) in enumerate(faces, start = 1):
                subset = image[x:x + w, y:y + h]
                cv2.imwrite(f"{idx}_{internal_idx}.png", subset)
        else:
            pass
            # cv2.imwrite(, image)
    except Exception as e:
        print(str(e))

char_name = "Eriri Spencer Sawamura"
files = [os.path.join(f"../anime_faces/{char_name}", x) for x in  os.listdir(f"../anime_faces/{char_name}")]

for idx, file in enumerate(files, start = 1):
    detect(file, idx)
