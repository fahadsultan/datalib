
from __future__ import print_function

import pandas as pd
import numpy as np
import json
from sklearn import cluster
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer
from sklearn.cluster import KMeans, MiniBatchKMeans, SpectralClustering
from random import randint
from scipy.spatial import distance
from scipy.sparse import vstack

from clustering import Clustering

from kde import KDE
from IPython.terminal.embed import InteractiveShellEmbed
ipshell = InteractiveShellEmbed()


class KMeans(Clustering):

	def get_seeds(self, data, col):
		
		"""
		Peforms KDE clustering on data and returns `col` of max density point for each cluster

		Parameters
		----------
		data : Pandas DataFrame
			A pandas dataframe atleast containing columns: lat, lng and `col`
		col : string or array of strings
			Column(s) to return for max density points of each cluster_number

		Returns 
		-------
		result : np.ndarray
			A 1-dimensional ndarray of `col` 
		"""

		if len(set(['lat', 'lng']+col).intersection(set(data.columns))) != 3:
			raise ValueError("One or more of columns: 'lat', 'lng' and %s not found.\
				 All three MUST be present" % col)

		# clusters = KDE().get_clusters(data[['lat', 'lng']], bandwidth=0.001, test_original=True)
		clusters = KDE().get_clusters(data[['lat', 'lng']], bandwidth=0.001)
		max_densities = clusters.groupby("cluster_number")['density'].max()
		max_density_points = []
		
		for i, density in max_densities.iteritems():

			row = clusters[(clusters['cluster_number'] == i) & (clusters['density'] == density)].iloc[0]

			distances = data.apply(lambda x: distance.euclidean((x['lat'], x['lng']), (row['lat'], row['lng'])), axis=1)
			# max_density_points = max_density_points.append(
			# 	data.iloc[distances[distances == distances.min()].index].iloc[0])

			max_density_points.append(distances[distances == distances.min()].index[0])

		return max_density_points

		# return max_density_points[col].as_matrix()

	def kmeans(self, data, features=None, text_features=[], n_clusters=8, centroid_features=10, random_seeds=True, 
		weights=[], use_idf=True):

		"""
		Applies KMeans clustering and returns labels and important features of each \
		cluster centroid

		Parameters
		----------
		data : Pandas DataFrame
			Data on which on apply clustering 
		features : list, optional, default : all columns used as features
			Subset of columns in the data frame to be used as features
		text_features : list, optional, default : None
			List of features that are of type text. These are then vectorizer using 
			TfidfVectorizer.
		n_clusters : int, optional, default: 8
			The number of clusters to form as well as the number of centroids to generate.
		centroid_features : int, optional, default: 10
			The number of most-important-features to return against each cluster centroid
		random_seeds : boolean, optional, default: False
			If False, uses clusters from kernel density estimation followed by thresholding
			as initial seeds. The number of clusters is also determined by results of kde and
			thus n_clusters parameter is ignored. 
		use_idf : boolean, optional, default: True
			If False, text_features are encoded using without idf 

		Returns
		-------
		result : tuple (labels, centroid_features)
			labels : 
				cluster numbers against each row of the data passed
			centroids : dictionary
				map of most important features of each cluster 
		"""

		X = self.encode_features(data, features, text_features, weights, use_idf)

		init = 'k-means++'
		# init = 'random'
		if random_seeds is False:
			seed_indices = self.get_seeds(data, features)
			init = vstack([X.getrow(seed_idx) for seed_idx in seed_indices]).todense()
			n_clusters = len(init)

		km = MiniBatchKMeans(n_clusters=n_clusters, init=init, n_init=5, init_size=10000,
		 batch_size=10000, verbose=True, tol=0.00001)
		# km = cluster.KMeans(n_clusters=n_clusters, init=init, n_init=5, verbose=True, tol=0.00001)
		km.fit(X)
		ipshell()

		print("Intertia: %s" % km.inertia_)

		centroids = {}

		order_centroids = km.cluster_centers_.argsort()[:, ::-1]

		if len(text_features) > 0:

			terms = self.vectorizer.get_feature_names()

			for i in range(n_clusters):
				important_features = ' '.join([terms[ind] if ind < len(terms) else str(i) for ind in order_centroids[i, :centroid_features]])
				centroids[i] = important_features
		else:
			for i in range(len(order_centroids)):
				centroids[i] = str(i)#str(order_centroids[i])

		return (km.labels_, centroids)


