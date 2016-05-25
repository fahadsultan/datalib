
import pandas as pd
from sklearn.neighbors import KernelDensity
import numpy as np
from scipy.ndimage import measurements
import math

import sys
sys.path.append('/Library/WebServer/Documents')

from Utils.utils import Utils

from IPython.terminal.embed import InteractiveShellEmbed
ipshell = InteractiveShellEmbed()

class KDE:

	def __init__(self):
		pass

	def get_matrix_from_two_arrs(self, x, y):

		xx, yy = np.meshgrid(x, y)
		return np.array(zip(xx.ravel(), yy.ravel()))

	def get_heatmap(self, points_df, kernel='gaussian', bandwidth=0.005,
			test_bl=None, test_tr=None, test_resolution=100, test_original=False):

		"""
		Returns estimation on specified test grid after using Kernel Density Estimation

		Parameters
		----------
		points_df : Pandas DataFrame
			An (n,2) dataframe containing lat, long points
		kernel : string
			The kernel to use.  Valid kernels are
		    ['gaussian'|'tophat'|'epanechnikov'|'exponential'|'linear'|'cosine']
		    Default is 'gaussian'.
		bandwidth : float
    		The bandwidth of the kernel.
		test_bl : tuple (lat, lng)
			bottom left corner of the matrix/grid to test on
			If None, defaults to min(lat), min(lng) values from points_df
		test_tr : tuple (lat, lng)
			top right corner of the matrix/grid to test on
			If None, defaults to max(lat), max(lng) values from points_df
		test_resolution: int
			Resolution of the test grid. Default is 100 (a 100*100 grid)
		test_original : boolean, optional, default False
			If True, test of original points; test_resolution, test_bl and 
			test_tr are ignored. 

		Returns
		-------
		result : np.array
			An (n,3) numpy array, with columns: lat, lng, density
		"""

		if test_original:
			test_grid = points_df.as_matrix()
		else:
			if (test_bl == None) or (test_tr == None):

				lat_col, lng_col = points_df.columns[0], points_df.columns[1]
				test_bl = points_df[lat_col].min(), points_df[lng_col].min()
				test_tr = points_df[lat_col].max(), points_df[lng_col].max()

			lat_test = np.linspace(test_bl[0], test_tr[0], test_resolution)[:, np.newaxis]
			lng_test = np.linspace(test_bl[1], test_tr[1], test_resolution)[:, np.newaxis]

			test_grid = self.get_matrix_from_two_arrs(lat_test, lng_test)

		kde = KernelDensity(kernel=kernel, bandwidth=bandwidth).fit(points_df.as_matrix())

		y = np.exp(kde.score_samples(test_grid))[:,np.newaxis]

		return np.hstack([test_grid, y])

	def get_heatmap_aggregated_over_col(self, points_df, col, kernel='gaussian', bandwidth=0.005,
			test_bl=None, test_tr=None, test_resolution=100):

		"""
		Returns a dataframe with densities values for aggregated `col`

		Parameters
		----------
		points_df : Pandas DataFrame
			An (n,3) dataframe containing columns 'lat', 'lng' and `col` param
		col : string 
			Column to aggregate on
		kernel : string
			The kernel to use.  Valid kernels are
		    ['gaussian'|'tophat'|'epanechnikov'|'exponential'|'linear'|'cosine']
		    Default is 'gaussian'.
		bandwidth : float
    		The bandwidth of the kernel.
		test_bl : tuple (lat, lng)
			bottom left corner of the matrix/grid to test on
			If None, defaults to min(lat), min(lng) values from points_df
		test_tr : tuple (lat, lng)
			top right corner of the matrix/grid to test on
			If None, defaults to max(lat), max(lng) values from points_df
		test_resolution: int
			Resolution of the test grid. Default is 100 (a 100*100 grid)

		Returns
		-------
		result : pandas.DataFrame
			DataFrame with columns 'lat', 'lng' followed by one column for each aggregation
		"""

		if len(set(points_df.columns).intersection(set([col, 'lat', 'lng']))) != 3:
			raise ValueError("'points_df' does not contain one or any of required columns: \
				lat, lng, %s" %col)

		response_df = None

		grobj = points_df.groupby(col)
		groups = grobj.groups.keys()
		groups.sort()

		for group in groups:
			group_data = grobj.get_group(group)
			heatmap = self.get_heatmap(group_data[['lat', 'lng']], kernel=kernel, bandwidth=bandwidth, 
				test_bl=test_bl, test_tr=test_tr, test_resolution=test_resolution)

			if response_df is None:
				response_df = pd.DataFrame(heatmap, columns=['lat', 'lng', group])
			else:
				response_df[group] = heatmap[:,2]

		return response_df

	def get_clusters(self, points_df, kernel='gaussian', bandwidth=0.005, test_bl=None, 
			test_tr=None, test_resolution=100, threshold=80, test_original=False):

		"""
		Thresholds a 3d map generated from KDE

		Parameters
		----------
		points_df : Pandas DataFrame
			An (n,2) dataframe containing columns 'lat', 'lng'
		kernel : string
			The kernel to use.  Valid kernels are
		    ['gaussian'|'tophat'|'epanechnikov'|'exponential'|'linear'|'cosine']
		    Default is 'gaussian'.
		bandwidth : float
    		The bandwidth of the kernel.
		test_bl : tuple (lat, lng)
			bottom left corner of the matrix/grid to test on
			If None, defaults to min(lat), min(lng) values from points_df
		test_tr : tuple (lat, lng)
			top right corner of the matrix/grid to test on
			If None, defaults to max(lat), max(lng) values from points_df
		test_resolution: int
			Resolution of the test grid. Default is 100 (a 100*100 grid)

		Returns
		-------
		result : pandas.DataFrame
			DataFrame with rows <= n and columns 'lat', 'lng', 'density', 'cluster_number'
		"""

		heatmap = self.get_heatmap(points_df, kernel=kernel, bandwidth=bandwidth, 
				test_bl=test_bl, test_tr=test_tr, test_resolution=test_resolution, 
				test_original=test_original)

		heatmap_df = pd.DataFrame(heatmap, columns=['lat', 'lng', 'density'])

		heatmap_df['thresholded'] = heatmap_df['density'] > np.percentile(heatmap_df['density'], threshold)

		if test_original:
			factors = list(Utils.factors(heatmap_df.shape[0]))
			factors.sort()
			rows = factors[math.floor(len(factors)/2)]
			cols = heatmap_df.shape[0]/rows
			test_resolution = (rows, cols)
		else:
			test_resolution = (test_resolution, test_resolution)

		X = heatmap_df['thresholded'][:,np.newaxis].reshape(test_resolution)

		cluster_numbers, num = measurements.label(X)

		heatmap_df['cluster_number'] = cluster_numbers.ravel()

		clusters = heatmap_df[heatmap_df['thresholded'] == True]

		del clusters['thresholded']

		return clusters


