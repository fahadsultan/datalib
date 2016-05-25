from kde import KDE
from scipy.interpolate import interp1d
from Utils.utils import Utils

import pandas as pd


from IPython.terminal.embed import InteractiveShellEmbed
ipshell = InteractiveShellEmbed()


class MultiVariateHeatmap:

	def __init__(self):
		pass

	def get_heatmap(self, variables_arr, kernel='gaussian', bandwidth=0.005,
			test_bl=None, test_tr=None, test_resolution=100):

		"""
		Returns estimation on specified test grid after using Kernel Density Estimation

		Parameters
		----------
		variables_arr : list of Pandas DataFrame objects
			Each dataframe should contain lat, long points and have the shape (n,2)
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
		result : np.array
			An (n,3) numpy array, with columns: lat, lng, hex-color-code
		"""

		if len(variables_arr) > 3: 
			raise NotImplementedError("As of now, only 3 variables are supported")

		kde = KDE()
		densities_arr = []

		for var in variables_arr:
			points_and_density = kde.get_heatmap(var, kernel=kernel, bandwidth=bandwidth, 
				test_bl=test_bl, test_tr=test_tr, test_resolution=test_resolution)

			density = points_and_density[:,2]
			mapping = interp1d([density.min(), density.max()], [0,255])
			densities_arr.append(mapping(density))

		ipshell()

		rgb_arr = zip(densities_arr[0], densities_arr[1], densities_arr[2])
		hex_arr = [Utils.rgb_to_hex(rgb) for rgb in rgb_arr]

		return pd.DataFrame().from_records(zip(points_and_density[:,0],points_and_density[:,1], hex_arr), columns=['lat', 'lng', 'color'])
