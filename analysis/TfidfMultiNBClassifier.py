
from sklearn.feature_extraction.text import CountVectorizer
from sklearn.feature_extraction.text import TfidfTransformer
from sklearn.naive_bayes import MultinomialNB

from Utils.utils import Utils

import numpy as np
import pandas as pd

from IPython.terminal.embed import InteractiveShellEmbed
ipshell = InteractiveShellEmbed()

class TfidfMultiNBClassifier: 

	def __init__(self):
		self.TEST_PERCENTAGE = 0.20

		self.count_vect = CountVectorizer()

	def train(self, X, y):

		# self.X 			= Utils.shuffle_sparse_matrix(X)
		self.X 			= X

		self.y 			= y

		X_counts 		= self.count_vect.fit_transform(self.X)
		self.X_tfidf 	= TfidfTransformer().fit_transform(X_counts)

		ipshell()

		self.last_training_index = int(round(self.TEST_PERCENTAGE*len(self.X)))

		X_train_tfidf 	= self.X_tfidf[:-self.last_training_index]
		y_train 		= self.y[:-self.last_training_index]

		self.clf 		= MultinomialNB().fit(X_train_tfidf, y_train)

	def test(self):

		X_test_tfidf  	= self.X_tfidf[-self.last_training_index:]
		y_test  		= self.y[-self.last_training_index:]

		predicted 		= self.clf.predict(X_test_tfidf)

		
		score = float(sum(predicted == y_test))/len(y_test)
		
		ipshell()
		
		return score

	def train_and_test(self, X, y, test_count=5000):

		self.train(X, y)
		return self.test()

