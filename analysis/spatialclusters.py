
import argparse
import pandas as pd
from sklearn.neighbors import KernelDensity
import numpy as np
from matplotlib import pyplot as plt
from random import randint
from scipy.ndimage import measurements
from scipy.spatial import ConvexHull, Delaunay, distance
from subprocess import call
from Utils.analysis.kde import KDE
from Utils.utils import Utils

from IPython.terminal.embed import InteractiveShellEmbed
ipshell = InteractiveShellEmbed()

class SpatialClusters:

	def __init__(self):
		pass

	def get_convex_hull(self, grp_df):

		try:
			# grp_df = grp_df.drop_duplicates()
			# grp_df['loc'] = grp_df['loc'].apply(ast.literal_eval)
			# points = np.array([point for point in list(grp_df['loc'])])

			points = grp_df[['lat', 'lng']].as_matrix()

			hull = ConvexHull(points, qhull_options="QJn")

			arr = []

			for simplex in hull.simplices:
				arr.append(points[simplex][0])
				arr.append(points[simplex][1])

			df = pd.DataFrame(arr, columns=['lat', 'lng'])

			df.index = range(len(df))
			df.loc[len(df)] = df.loc[0]

			return_obj = df.to_json(orient='records')

		except Exception as e:
			return_obj = {}


		print "******************"
		print return_obj
		return return_obj

	def get_neighboring_indices(self, x, y, X, Y, four_connected=True): 
		"""
			four_connected : Boolean 
				If True, returns four neighbors, otherwise eight
		"""

		if four_connected:

			neighbors = []

			left, right, top, bottom = x-1, x+1, y-1, y+1

			if left >= 0:
				neighbors.append([left, y])
			if top >= 0:
				neighbors.append([x, top])
			if right < X:
				neighbors.append([right, y])
			if bottom < Y:
				neighbors.append([x, bottom])
		else:
			neighbors = [(x2, y2) for x2 in range(x-1, x+2)
							   for y2 in range(y-1, y+2)
							   if (-1 < x <= X and
								   -1 < y <= Y and
								   (x != x2 or y != y2) and
								   (0 <= x2 <= X) and
								   (0 <= y2 <= Y))]


		return neighbors

	def identify_boundary_points(self, dataframe):

		is_boundary_point = []

		mat = dataframe['thresholded'].as_matrix()

		mat = mat.reshape(100,100)

		X, Y = mat.shape[0]-1, mat.shape[1]-1

		for index, item in np.ndenumerate(mat):	

			if item == False:
				is_boundary_point.append(False)
				continue

			neighbor_indices = self.get_neighboring_indices(index[0], index[1], X, Y)

			neighbor_vals = [mat[n_index[0], n_index[1]] for n_index in neighbor_indices]

			is_boundary_point.append(np.any(np.invert(neighbor_vals)))

		return is_boundary_point

	def get_heatmap(self, latlng_arr, bandwidth=0.001):
		data = pd.DataFrame(latlng_arr, columns=['Latitude','Longitude'])
		data = data[data['Latitude'] != 0]

		kde = KDE()
		points_and_density = kde.get_heatmap(data)

		return pd.DataFrame(points_and_density, columns=['lat', 'lng', 'density'])

	def get_polygon(self, heatmap):

		cols = list(heatmap.columns)

		heatmap_m = heatmap.as_matrix().reshape(100,100, len(cols))
		
		cluster_numbers = np.delete(np.unique(heatmap_m[:,:, cols.index('cluster_number')]), 0)

		polygons = {}

		polygons_df = pd.DataFrame(columns=['cluster_number', 'boundary'])

		for cluster_no in cluster_numbers:

			ordered_list = []

			xx, yy = np.where((heatmap_m[:,:,cols.index('cluster_number')] == cluster_no) & (heatmap_m[:,:,cols.index('is_boundary_point')]))

			boundary_points = np.array(zip(xx.ravel(), yy.ravel()))
	
			curr_ind = boundary_points[0]
			
			while True:

				ordered_list.append(tuple(curr_ind))

				distances = np.array([distance.euclidean(curr_ind, point) for point in boundary_points])
				# neighbors_and_boundary = boundary_points[np.where((distances < 2) & (distances != 0))]
				neighbors_and_boundary = np.concatenate([boundary_points[np.where((distances <= 1) & (distances != 0))], boundary_points[np.where((distances < 2))]])

				found = False
				for i, n_and_b in enumerate(neighbors_and_boundary):

					if tuple(n_and_b) not in ordered_list:
						curr_ind = n_and_b

						bool_map = [np.all(i) for i in (boundary_points == n_and_b)]
						boundary_points = np.delete(boundary_points, bool_map.index(True), 0)

						found = True
						break


				if found is False:

					bool_map = [np.all(i) for i in (boundary_points == curr_ind)]
					if sum(bool_map) > 0:
						boundary_points = np.delete(boundary_points, bool_map.index(True), 0)

					if len(boundary_points) > 0:
						# polygons[cluster_no] = [{'lat':heatmap_m[point][0], 'lng':heatmap_m[point][1]} for point in ordered_list]
						boundary = [{'lat':heatmap_m[point][0], 'lng':heatmap_m[point][1]} for point in ordered_list]
						boundary = self.smooth_out_straight_lines(boundary)
						polygons_df.loc[len(polygons_df)] = {'cluster_number':cluster_no, 'boundary':boundary}
						ordered_list = []
						curr_ind = boundary_points[0]
						# closest_ind = np.where(distances == distances[distances != 0].min())[0][0]
						# curr_ind = boundary_points[closest_ind]
						# boundary_points = np.delete(boundary_points, closest_ind, 0)
					else:
						# polygons[cluster_no] = [{'lat':heatmap_m[point][0], 'lng':heatmap_m[point][1]} for point in ordered_list]
						boundary = [{'lat':heatmap_m[point][0], 'lng':heatmap_m[point][1]} for point in ordered_list]
						boundary = self.smooth_out_straight_lines(boundary)
						polygons_df.loc[len(polygons_df)] = {'cluster_number':cluster_no, 'boundary':boundary}
						break

		return polygons_df

	def smooth_out_straight_lines(self, boundary):

		# ipshell()
		if len(boundary) < 3:
			return boundary

		def slope(point_one, point_two):
			lat_diff = (point_two['lat'] - point_one['lat'])
			lng_diff = (point_two['lng'] - point_one['lng'])
			# return  (lat_diff / lng_diff) if (lng_diff != 0) else 9999
			return (round(lat_diff, 4), round(lng_diff, 4))

		points_to_remove = []

		for i in range(1,len(boundary) - 1):
			first_slope = slope(boundary[i-1], boundary[i])
			second_slope = slope(boundary[i], boundary[i+1])

			if first_slope == second_slope:
				points_to_remove.append(boundary[i])

		for points_to_remove in points_to_remove:
			boundary.remove(points_to_remove)

		return boundary

	def get_clusters(self, latlng_arr, bandwidth, threshold=80, triangulate=False):
		"""
		Returns convex hulls of spatial clusters. 

		Parameters:
		----------

		latlng_arr : array of latitude and longitude values 
		threshold  : float or int in range of [0,100]. Percentile at which to apply threshold. 
		"""

		heatmap = self.get_heatmap(latlng_arr, bandwidth)

		heatmap['thresholded'] = heatmap['density'] > np.percentile(heatmap['density'], threshold)#0.01

		heatmap['is_boundary_point'] = self.identify_boundary_points(heatmap)

		X = heatmap['thresholded'][:,np.newaxis].reshape(100,100)
		cluster_numbers, num = measurements.label(X)
		heatmap['cluster_number'] = cluster_numbers.ravel()

		clusters = heatmap[heatmap['thresholded'] == True]
		print "NUMBER OF POINTS: %s" % len(clusters)

		# hulls = clusters.groupby('cluster_number').apply(self.get_convex_hull)

		# hulls = clusters.groupby('cluster_number').apply(lambda x: x[x['thresholded']].to_json(orient='records'))

		polygons = self.get_polygon(heatmap)

		if triangulate:
			self.triangulate(polygons)

		return clusters, polygons

	# def triangulate(polygons_df):

	# 	vis_data = pd.DataFrame(columns=['cluster_number', 'boundary'])

	# 	for idx, row in polygons_df.iterrows():
	# 		boundary = np.array(pd.DataFrame(row['boundary']))
	# 		if len(boundary) < 3:
	# 			continue
	# 		delauney = Delaunay(boundary)
	# 		simplices = delauney.simplices
	# 		for simplice in simplices:
	# 			tri_points = pd.DataFrame(boundary[simplice], columns=['lat', 'lng']).to_json(orient='records')
	# 			vis_data.loc[len(vis_data)] = {'cluster_number':row['cluster_number'], 'boundary':tri_points}


	# 	colors_df = pd.DataFrame()
	# 	cluster_numbers = range(len(vis_data['cluster_number'].unique()))
	# 	colors_df['cluster_number'] = cluster_numbers
	# 	colors_df['colors'] = ['#%06X' % randint(0, 0xFFFFFF) for i in cluster_numbers]

	# 	# hulls = pd.DataFrame(hulls)
	# 	# hulls['cluster_number'] = range(len(hulls))

	# 	merged = pd.merge(vis_data, colors_df, on='cluster_number')

	# 	merged.to_csv('triangles.csv')

	def triangulate(self, polygons_df):
		
		arr = []
		for i, row in polygons_df.iterrows():
			in_f = open("lib/triangle/poly_out.poly", "w")
			boundaries = pd.DataFrame(polygons_df.iloc[i]['boundary'])
			if len(boundaries) < 3:
				continue

			in_f.write(str(len(boundaries))+" 2 0 0\n")
			in_f.write(pd.DataFrame(boundaries).to_csv(sep=' ', header=False))
			in_f.write(str(len(boundaries))+" 0\n")
			df = pd.DataFrame()
			df['source'] = range(len(boundaries))
			df['destination'] = df['source'].shift(-1).fillna(0).astype('int')
			in_f.write(df.to_csv(sep=' ', header=False))
			in_f.write('0')
			in_f.close()
			
			call(["lib/triangle/triangle", "-p", "lib/triangle/poly_out.poly"])

			with open("lib/triangle/poly_out.1.ele") as out_f:
				lines = out_f.readlines()[1:-1]

			lines = [re.sub( '\s+', ' ', line).strip().split() for line in lines]

			for line in lines:
				try:
					arr.append([dict(boundaries.loc[int(line[1])]),
						dict(boundaries.loc[int(line[2])]),
						dict(boundaries.loc[int(line[3])])])
				except Exception as e:
					continue

		print "NUMBER OF TRIANGLES: %s" % len(arr)
		pd.DataFrame(pd.Series(arr)).to_csv('out/data/triangles2.csv', index=False)

def get_heatmap(data, outpath, bandwidth):
	sc = SpatialClusters()
	heatmap = sc.get_heatmap(data[['lat', 'lng']].dropna().as_matrix(), bandwidth)
	heatmap['colors'] = Utils.assign_colors_o(heatmap['density'], quantiles=[0.6, 0.75, 0.9, 0.95, 0.99])
	heatmap.to_csv(outpath)

def get_clusters(data, outpath, hullpath, bandwidth, threshold, hull_color=None):
	sc = SpatialClusters()
	clusters, hulls = sc.get_clusters(data[['lat', 'lng']].dropna().as_matrix(), bandwidth, threshold)
	clusters['colors'] = "#d73027"
	clusters['colors'][clusters['is_boundary_point']] = "#ffff00"

	if outpath is not None:
		clusters.to_csv(outpath)

	if hullpath is not None:

		if hull_color is None:
			colors_df = pd.DataFrame()
			colors_df['cluster_number'] = range(len(hulls))
			colors_df['colors'] = ['#%06X' % randint(0, 0xFFFFFF) for i in range(len(hulls))]
			hulls = pd.merge(hulls, colors_df, on='cluster_number')
		else:
			hulls['colors'] = hull_color

		hulls.to_csv(hullpath)
