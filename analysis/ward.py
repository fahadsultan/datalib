

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

from scipy.cluster.hierarchy import ward, dendrogram, fcluster
from sklearn.metrics.pairwise import cosine_similarity

from clustering import Clustering

from kde import KDE
from IPython.terminal.embed import InteractiveShellEmbed
ipshell = InteractiveShellEmbed()


class Agglomerative(Clustering):

	def __dir__(self):
		return self.keys()

	# def get_k_clusters(self):

	# 	linkages = pd.DataFrame(self.linkage_matrix, columns=[
	# 		'cluster_a', 'cluster_b', 'distance', 'number_of_nodes'])

	# 	linkages = linkages.reindex(index=linkages.index[::-1])

	# 	for row in linkages.iterrows():


		

	def get_clusters(self, data, features=None, text_features=[], n_clusters=8, centroid_features=10, random_seeds=True, 
		weights=[]):

		"""
		Applies Agglomerative hierarchial clustering using Ward's linkage

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

		Returns
		-------
		result : tuple (labels, centroid_features)
			labels : 
				cluster numbers against each row of the data passed
			centroids : dictionary
				map of most important features of each cluster 
		"""

		X = self.encode_features(data, features, text_features)

		ipshell()

		dist = 1 - cosine_similarity(X)

		self.linkage_matrix = ward(dist)

		return (km.labels_, centroids)

