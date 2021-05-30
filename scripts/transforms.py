from fastai.vision.all import *
import albumentations

def encode_image(x, train = True):
    
    # Open the image and conveet it to a Tensor
    image = np.array(Image.open(x).convert('RGB').copy())
    
    if train:
        # Apply albumentation transforms
        tfms = albumentations.Compose([
                                        albumentations.Resize(300, 300),
                                        albumentations.RandomResizedCrop(225, 225),
                                        albumentations.Transpose(p=0.5),
                                        albumentations.VerticalFlip(p=0.5),
                                        albumentations.ShiftScaleRotate(p=0.5),
                                        albumentations.HueSaturationValue(
                                            hue_shift_limit=0.2, 
                                            sat_shift_limit=0.2, 
                                            val_shift_limit=0.2, 
                                            p=0.5),
                                        albumentations.CoarseDropout(p=0.5),
                                        albumentations.Cutout(p=0.5),
                                      ])
        transformed_image = tfms(image = image)
    else:
        tfms = albumentations.Compose([
                                        albumentations.Resize(225,225)
                                      ], p=1.)
        transformed_image = tfms(image = image)
        

    # Normalize the image
    norm_image = albumentations.normalize(transformed_image['image'],
                                          mean = imagenet_stats[0],
                                          std = imagenet_stats[1])
    
    # Transpose the image to have channels first
    norm_image = torch.tensor(norm_image).permute(2, 0, 1)
    
    # Build a tensor out of the normed image
    tensor_image = TensorImage(norm_image)

    return tensor_image

class trainTransform(Transform):
    def encodes(self, x):
        op = encode_image(x, True)
        return op

class validTransform(Transform):
    def encodes(self, x):
        op = encode_image(x, False)
        return op