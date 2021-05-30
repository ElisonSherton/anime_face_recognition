from fastai.vision.all import *
import albumentations
import PIL.Image

def read_image(pth):
    '''
    Given the path to an image, read it normalize it and tensorify it.
    Create a one image batch of the resultant tensor and return the same
    '''
    
    # Open the image
    i = np.array(PIL.Image.open(pth).convert('RGB'))
    
    # Resize the image
    resized_img = albumentations.Resize(225, 225)(image = i)['image']
    
    # Normalize the image
    norm_img = albumentations.normalize(resized_img, mean = imagenet_stats[0], std = imagenet_stats[1])
    
    # Convert the image into a torch tensor
    tensor_img = torch.tensor(norm_img).permute(2, 0, 1)
    
    # Batchify the image
    tensor_img = tensor_img.unsqueeze(0)
    
    return tensor_img

def get_fv(pth, model):
    '''
    Get the feature vector of an image given the model and the image path
    '''
    # Read in the image as a tensor
    tensor_image = read_image(pth)
    
    # Put the model in evaluation mode
    model.eval()
    
    with torch.no_grad():
        # Extract the embedding by doing a forward pass and get the activations in numpy
        embedding = model(tensor_image).cpu().numpy()[0]
    
    return embedding