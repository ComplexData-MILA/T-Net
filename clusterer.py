'''
Author: Pratheeksha Nair

This file contains the implementation of different clustering methods

'''

from sklearn.cluster import KMeans
from kmeans_pytorch import kmeans
from sklearn.metrics.pairwise import euclidean_distances, cosine_distances
import torch

class Clusterer():
	def _return_dists_from_centroid(self, data):
		centroids = self.kmeans.cluster_centers_
		cluster_labels = self.kmeans.labels_
		dist_to_closest_centroid = []
		for i, node in enumerate(data):
			curr_clus = cluster_labels[i]
			curr_clus_center = centroids[curr_clus]
			# dd = euclidean_distances(node.reshape(1, -1), curr_clus_center.reshape(1, -1)).item()
			dd = cosine_distances(node.reshape(1, -1), curr_clus_center.reshape(1, -1)).item()
			dist_to_closest_centroid.append(dd)
		
		return dist_to_closest_centroid


	def cluster_kmeans(self, data, num_clusters):
		# print(data)
		self.kmeans = KMeans(n_clusters=num_clusters, random_state=0, n_init=10).fit(data)

	def _return_labels(self):
		return torch.Tensor(self.kmeans.labels_)

	def cluster(self, data, num_clusters, device='cpu'):
		# kmeans
		cluster_ids_x, cluster_centers = kmeans(
		    X=data, num_clusters=num_clusters, distance='euclidean', iter_limit=100, device=torch.device(device)
		)
		# clusters = self.method.fit(data).labels_
		return cluster_ids_x

	def create_louvain_clusterer(self):
		pass
		
	def __init__(self, name):
		if name == 'Louvain':
			self.method = create_louvain_clusterer()
		elif name == 'Kmeans':
			self.method = 'kmeans'