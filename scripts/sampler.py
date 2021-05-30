from fastai.vision.all import *
import random

def sampler(df, batch_size = 16, k = 4):
    # Calculate the number of images per class
    p = batch_size // k
    
    # Find out the total number of samples and the number of batches
    n = len(df)
    n_batches = n // batch_size
    
    # Compute a mapping of class -> items 
    class_item_mapping = {}
    for idx, row in enumerate(df.itertuples()):
        c_name = row[1].split('/')[0]
        if c_name in class_item_mapping:
            class_item_mapping[c_name].append([row[1], idx])
        else:
            class_item_mapping[c_name] = [[row[1], idx]]
    
    classes = list(class_item_mapping.keys())
    
    # Create an index list by randomly sampling n_batches batches with k classes & p images per batch
    indices = []
    
    for i in range(n_batches):
        # Sample k classes for every batch
        cls = random.sample(classes, k)
        
        batch_indices = []
        
        # Sample p images per class
        for c in cls:
            c_samples = random.sample(class_item_mapping[c], p)
            c_indices = [x[1] for x in c_samples]
            batch_indices.extend(c_indices)
        
        indices.extend(batch_indices)
    
    return indices

