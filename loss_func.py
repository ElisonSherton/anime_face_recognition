# https://arxiv.org/pdf/1703.07737.pdf
import torch
import torch.nn as nn

class batchHardTripletLoss(nn.Module):
    def __init__(self, margin = 1, squared = False, agg = "mean"):
        """
        Initialize the loss function with a margin parameter, whether or not to consider
        squared Euclidean distance and how to aggregate the loss in a batch
        """
        super(batchHardTripletLoss, self).__init__()
        self.margin = margin
        self.squared = squared
        self.agg = agg
        self.eps = 1e-8
    
    def get_pairwise_distances(self, feat_vecs):
        """
        Computing distance for every pair using 
        (a - b) ^ 2 = a^2 - 2ab + b^2 
        """
        ab = feat_vecs.mm(feat_vecs.t())
        a_squared = ab.diag().unsqueeze(1)
        b_squared = ab.diag().unsqueeze(0)
        distances = a_squared - 2 * ab + b_squared
        distances = nn.ReLU()(distances)
        
        if not self.squared:
            distances = torch.sqrt(distances + self.eps)

        return distances
            
        
    def get_mask(self, labels, type_ = "positive"):
        """
        Get a binary matrix corresponding to valid duplet pairs for 
        (anchor, positive) & (anchor, negative) pairs
        """
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"
        PK = labels.shape[1]
        mask = torch.zeros(PK, PK).to(DEVICE)
        
        for idx, item in enumerate(labels[0]):
            for inner_idx, inner_item in enumerate(labels[0]):
                
                if type_ == "positive":
                    
                    # Labels should match and the image index shouldn't be the same
                    if (item == inner_item) and (idx != inner_idx):
                        mask[idx, inner_idx] = 1
                elif type_ == "negative":
                    
                    # Labels must be different and image index shouldn't be the same (redundant but still...)
                    if (item != inner_item) and (idx != inner_idx):
                        mask[idx, inner_idx] = 1
                    
        return mask
    
    def forward(self, feat_vecs, labels):
        """
        Define the loss function implementation here
        """
        DEVICE = "cuda" if torch.cuda.is_available() else "cpu"

        # Get the pairwise distances of all images from one another
        distances = self.get_pairwise_distances(feat_vecs)
        
        # Get the toughest positive pair by first filtering out the (anchor, positive)
        # pairs using the get_mask routine and then find the max across rows
        positive_mask = self.get_mask(labels, type_ = "positive").to(DEVICE)
        toughest_positive_distance = (distances * positive_mask).max(dim = 1)[0]
        
        negative_mask = self.get_mask(labels, type_ = "negative").to(DEVICE)
        
        # Add the maxiumum negative distance to all the non-valid pairs
        # on a rowwise basis and then out of them whichever is the minimum 
        # will be our pair distance corresponding to toughest (anchor, negative) pair
        max_negative_dist = distances.max(dim=1,keepdim=True)[0]
        distances = distances + max_negative_dist * (1 - negative_mask).float()
        toughest_negative_distance = distances.min(dim = 1)[0]
        
        # Find the triplet loss by using the two distances obtained above
        triplet_loss = nn.ReLU()(toughest_positive_distance - toughest_negative_distance + self.margin)
        
        # Aggregate the loss to mean/sum based on the initialization of the loss function
        if self.agg == "mean":
            triplet_loss = triplet_loss.mean()
        elif self.agg == "sum":
            triplet_loss = triplet_loss.sum()
            
        return triplet_loss