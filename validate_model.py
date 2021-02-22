import numpy as np
import pickle
from sklearn.metrics import pairwise_distances

# Mention the path to the extracted feature vectors here
FV_PATH = "/home/vinayak/anime_feature_vectors_valid.pkl"

with open(FV_PATH, "rb") as f:
    data = pickle.load(f)
    f.close()

# Extract feature vectors and calculate pairwise distances of every vector from every other vector
files = list(data.keys())
fvs = np.array(list(data.values()))
distances = pairwise_distances(fvs, n_jobs = -1)

# A function to compute the topk accuracy given a distance matrix and the lab
def compute_topk(distance_matrix, filenames, k = 1):
    
    get_label = lambda x: filenames[x].split("/")[-2]
    count = 0
    
    # Find the k nearest neighbours for every query image in the distance matrix
    for idx, vector in enumerate(distance_matrix):
        ranking = np.argsort(vector)
        indices = ranking[1:(k + 1)]
        
        source_label = get_label(idx)
        predicted_labels = [get_label(x) for x in indices]
        
        # If the neighbours have the same label as the source image, increment the count by 1
        if source_label in predicted_labels:
            count += 1
    
    # Compute % accuracy by counting the matches and taking it's ratio with the total present images
    top_k_accuracy = 100 * (count * 1.0) / distances.shape[0]
    return round(top_k_accuracy, 5)

# Print the top 1 - top 10 accuracy to see how our model performs
for k in range(1, 11):
    print(f"Top {k:<2} accuracy: {compute_topk(distances, files, k):<7}%")