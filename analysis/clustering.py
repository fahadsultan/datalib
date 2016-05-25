
from __future__ import print_function

import numpy as np
import scipy
from scipy import sparse as sp
from scipy.sparse import csr_matrix
from sklearn import preprocessing
from sklearn.feature_extraction.text import TfidfVectorizer

from IPython.terminal.embed import InteractiveShellEmbed
ipshell = InteractiveShellEmbed()

class Clustering(object):

	def __dir__(self):
		return self.keys()

	def encode_features(self, data, features, text_features, weights, use_idf=True):

		"""
		
		Encodes features amd returns a matrix of type numpy array 

		Parameters
		----------
		data : Pandas DataFrame
			Data from which to draw features from
		features : list, optional, default : all columns used as features
			Subset of columns in the data frame to be used as features
		text_features : list, optional, default : None
			List of features that are of type text. These are then vectorizer using 
			TfidfVectorizer.
		use_idf : boolean, optional, default: True
			If False, text_features are encoded using without idf 

		Returns 
		-------
		result : encoded matrix of type numpy array 

		"""
		X = None
		if 	(features is not None) and (
			set(data.columns).intersection(set(features)) != set(features)):
				raise ValueError("One or more of features not found in the dataframe passed")

		if (len(text_features) > 0) and (
			set(features).intersection(set(text_features)) != set(text_features)):
				raise ValueError("One or more of text_features not found in features")

		for txt_feature in text_features:

			self.vectorizer = TfidfVectorizer(max_df=0.5, min_df=2, stop_words='english', use_idf=use_idf)
			tfidf = self.vectorizer.fit_transform(list(data[txt_feature]))
			# X = np.hstack([X, tfidf.todense()]) if X is not None else tfidf.todense()
			X = sp.hstack([csr_matrix(X), tfidf]) if X is not None else tfidf

		for feature in set(features).difference(set(text_features)):
			new_feature = data[feature].as_matrix()[:,np.newaxis]
			X = sp.hstack([X, csr_matrix(new_feature)]) if X is not None else csr_matrix(new_feature)
			# X = np.hstack([X, new_feature]) if X is not None else new_feature
			if feature in weights.keys():
				for i in range(weights[feature]-1):
					print("encoding feature, weight: %s" % i)
					X = sp.hstack([X, csr_matrix(new_feature)])
					# X = np.hstack([X, new_feature])

		X_scaled = preprocessing.scale(X, with_mean=False)
		
		return X_scaled

