

from __future__ import print_function

import pandas as pd
import numpy as np
import json
from sklearn import cluster, manifold, preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering
from sklearn.cluster import spectral
from random import randint
from scipy.spatial import distance
from scipy.sparse import vstack

from matplotlib import pyplot as plt

from clustering import Clustering

from sklearn.metrics.pairwise import euclidean_distances

from kde import KDE
from IPython.terminal.embed import InteractiveShellEmbed
ipshell = InteractiveShellEmbed()

class SpectralClustering(Clustering):

	def fit(self, data, features=None, text_features=[], n_clusters=8, weights=[], use_idf=True):

		self.data = data
		self.features = features
		self.text_features = text_features

		self.X = self.encode_features(data, features, text_features, weights, use_idf)

	def transform(self, n_clusters=8, random_seeds=True):		

		self.n_clusters = n_clusters
		self.random_seeds = True

		self.spectral_embedding = manifold.SpectralEmbedding()
		self.Y = self.spectral_embedding.fit(self.X)

		self.spectral_clustering = spectral.SpectralClustering(n_clusters = self.n_clusters, 
			affinity='precomputed')

		self.spectral_clustering.fit(self.spectral_embedding.affinity_matrix_)

		ipshell()

		return self.spectral_clustering.labels_

	def get_most_important_features(self, top_n=10):

		features = self.vectorizer.get_feature_names()

		results = {}

		for i in range(self.n_clusters):

			cluster_points = self.X[self.spectral_clustering.labels_ == i].todense()
			indices = np.argsort(sum(cluster_points)).tolist()[0][::-1]

			top_features = ' '.join([features[j] if j < len(features) else ' ' for j in indices[:top_n]])
			results[i] = top_features

		return results

	def compute_sum_squared_error(self):
		sum_squared_error = 0

		for i in range(self.n_clusters):

			print("i: %s n_clusters: %s" % (i, self.n_clusters))
			
			cluster_points = self.X[self.spectral_clustering.labels_ == i].todense()

			centroid = cluster_points.mean(0)

			sum_squared_error = sum_squared_error + sum(euclidean_distances(cluster_points, centroid)**2)[0]

		return sum_squared_error

	def elbow_method(self, lower_k, higher_k, outpath):

		results = []

		for k in range(lower_k, higher_k+1):

			self.transform(n_clusters=k)
			squared_error = self.compute_sum_squared_error()

			results.append((k,squared_error))

		results = np.array(results)
		plt.plot(results[:,0], results[:,1])
		plt.savefig(outpath)




