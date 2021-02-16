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
    
    def forward(self, anchor_batch, positive_batch, negative_batch):
        
        # Function to transpose the vector in a shape suitable for model forward propagation
        trp = lambda x: torch.transpose(torch.transpose(x, 2, 3), 1, 2).float()

        # Map the transpose function to all anchor, positive and negative batch of images
        a, p, n = map(trp, (anchor_batch, positive_batch, negative_batch))
        
        # Pass the batches manipulted above through our model
        a, p, n = self.eff(a), self.eff(p), self.eff(n)
        
        # Return the feature vectors extracted from image batches
        return (a, p, n)