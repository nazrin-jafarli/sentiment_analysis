import torch
from scipy.spatial.distance import euclidean
# from sklearn.metrics.pairwise import cosine_similarity as sk_cosine_similarity
from scipy.spatial.distance import cosine as scipy_cosine_similarity
import numpy as np

class KMeansSemiSupervised:
    def __init__(self, n_clusters=3, max_iter=10, distance_measure='cosine'):
        self.n_clusters = n_clusters
        self.max_iter = max_iter
        self.centroids = None
        self.distance_measure = distance_measure
        self.unlabeled_embeddings = None
        self.centroid_labels = None  # Add centroid_labels attribute


    def initialize_centroids(self, labeled_data, labels):
        if len(labeled_data) > 0:
            self.centroids = {}
            self.centroid_labels = {}
            unique_labels = torch.unique(labels)
            for idx, label_value in enumerate(unique_labels):
                label_data = labeled_data[labels == label_value]
                centroid = label_data.mean(dim=0)
                self.centroids[idx] = centroid
                self.centroid_labels[idx] = label_value.item()
        else:
            self.initialize_centroids_randomly()


    def initialize_centroids_randomly(self):
        random_indices = torch.randperm(len(self.unlabeled_embeddings))[:self.n_clusters]
        self.centroids = {i: self.unlabeled_embeddings[index] for i, index in enumerate(random_indices)}
        self.centroid_labels = {i: None for i in range(self.n_clusters)}


    # fit function in which centroids update consider both labelled and unlabelled data
        
    def fit(self, labeled_embeddings, unlabeled_embeddings, labels):
        self.unlabeled_embeddings = unlabeled_embeddings
        self.initialize_centroids(labeled_embeddings, labels)

        all_embeddings = torch.cat((labeled_embeddings, unlabeled_embeddings), dim=0)


        # print(self.distance_measure)
        centroid_list = [centroid.cpu().numpy().flatten() for centroid in self.centroids.values()]  # Ensure centroids are 1-D

        for _ in range(self.max_iter):
            nearest_centroid_indices = []
            for x in all_embeddings:
                distances = []
                if self.distance_measure == 'cosine':
                    x_flattened = x.cpu().numpy().flatten()  # Ensure x is 1-D
                    similarities = [1 - scipy_cosine_similarity(x_flattened, centroid) for centroid in centroid_list]
                    distances = similarities

                    
                elif self.distance_measure == 'euclidean':
                    distances = [euclidean(x.cpu().numpy().flatten(), centroid) for centroid in centroid_list]

                nearest_centroid_indices.append(torch.argmin(torch.tensor(distances)).item())

            nearest_centroid_indices = torch.tensor(nearest_centroid_indices)

            for cluster_idx in range(self.n_clusters):
                cluster_indices = (nearest_centroid_indices == cluster_idx).nonzero().flatten()
                cluster_embeddings = all_embeddings[cluster_indices]
                if len(cluster_embeddings) > 0:
                    self.centroids[cluster_idx] = cluster_embeddings.mean(dim=0)

        return self.centroids


    def predict(self, unlabeled_embeddings):
        cluster_indices = []
        centroid_list = [centroid.cpu().numpy().flatten() for centroid in self.centroids.values()]  # Ensure centroids are 1-D

        for x in unlabeled_embeddings:
            distances = []
            for centroid in centroid_list:
                if self.distance_measure == 'cosine':
                    x_flattened = x.cpu().numpy().flatten()  # Ensure x is 1-D
                    similarity = 1 - scipy_cosine_similarity(x_flattened, centroid)
                    distance = similarity
                elif self.distance_measure == 'euclidean':
                    distance = euclidean(x.cpu().numpy().flatten(), centroid)
                distances.append(distance)
            closest_centroid_index = np.argmin(distances)
            semantic_label = self.centroid_labels[closest_centroid_index]
            cluster_indices.append(semantic_label)
        return cluster_indices

