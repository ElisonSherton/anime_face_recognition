import torch
from torch import nn
import timm

class enet_model(nn.Module):
    def __init__(self):
        
        super().__init__()
        
        # Create a model with pretrained weights
        model = timm.create_model("efficientnet_b3a", pretrained = True)
        
        # Extract all but last 2 layers of the model and add a Average pooling and Flatten layer 
        # to obtain 1-D feature vector for our images
        children = list(model.children())[:-2]
        children.extend([nn.AdaptiveAvgPool2d(1),nn.Flatten()])
        
        # Create a instance variable for the model obtained by binding all the child layers above
        self.eff = nn.Sequential(*children)
    
    def forward(self, image_batch):        
        # Return the feature vectors extracted from image batches
        return self.eff(image_batch)