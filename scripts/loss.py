from fastai.vision.all import *
from pytorch_metric_learning import miners, losses

class loss(nn.Module):
    def __init__(self, margin = 0.1, agg = 'mean'):
        '''
        Initialize the basic components of the loss function
        1. margin -> What is the margin when computing the loss = d(A, P) - d(A, N) + m; m = margin
        2. miner -> Mine batch hard triplets as described in the paper 
           https://arxiv.org/abs/1703.07737
        3. loss_func -> The actual loss computation function
        '''
        super().__init__()
        self.margin = margin
        self.miner = miners.MultiSimilarityMiner()
        self.loss_func = losses.TripletMarginLoss()
    
    def forward(self, embeddings, labels):
        '''
        Define the working of the loss function here
        '''
        hard_pairs = self.miner(embeddings, labels)
        loss = self.loss_func(embeddings, labels, hard_pairs)
        return loss