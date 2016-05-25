
from kde import KDE
from scipy.stats import ks_2samp
from matplotlib import pyplot as plt


from IPython.terminal.embed import InteractiveShellEmbed
ipshell = InteractiveShellEmbed()

class KSTest:

	def compare(self, dist_one, dist_two, plot_path=None, 
			dist_one_label="Dist_One", dist_two_label="Dist_Two", resolution=50):

		"""
		Performs KS-test

		Parameters:
		-----------

		dist_one : pandas DataFrame 

		dist_two : pandas DataFrame

		plot_path : String 

		dist_one_label : String 

		dist_two_label : String 

		resolution : int 

		Returns:
		--------

		ks_statistic

		p_value 

		"""

		min_lat = min(dist_one['lat'].min(), dist_two['lat'].min())
		max_lat = max(dist_one['lat'].max(), dist_two['lat'].max())

		min_lng = min(dist_one['lng'].min(), dist_two['lng'].min())
		max_lng = max(dist_one['lng'].max(), dist_two['lng'].max())

		test_bl = (min_lat, min_lng)
		test_tr = (max_lat, max_lng)

		y_one = KDE().get_heatmap(dist_one, test_resolution=resolution, test_bl=test_bl, test_tr=test_tr)[:,2]
		
		y_two = KDE().get_heatmap(dist_two, test_resolution=resolution, test_bl=test_bl, test_tr=test_tr)[:,2]

		ks_statistic, pvalue = ks_2samp(y_two, y_one)

		# ipshell()

		if plot_path is not None:

			ax = plt.subplot(211, title="Number of %s: %s\nNumber of %s: %s\nKS-Statistic: %s\nP-Value: %s "\
				 %(dist_one_label, len(dist_one), dist_two_label, len(dist_two), ks_statistic, pvalue))

			plt.pcolor(y_one.reshape(resolution,resolution))

			ax = plt.subplot(212, title="%s(above) %s(below)" %(dist_one_label, dist_two_label))

			plt.pcolor(y_two.reshape(resolution,resolution))

			plt.tight_layout()

			plt.savefig(str(plot_path))

		return ks_statistic, pvalue
		
