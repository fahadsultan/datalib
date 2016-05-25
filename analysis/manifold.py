
from sklearn import manifold 
from matplotlib import pyplot as plt

def try_all_manifolds(X, n_components, n_neighbors, outpath):

	ax = plt.subplot(311)

	lle_methods = []#['standard', 'ltsa', 'modified'] #'hessian',

	for i, method in enumerate(lle_methods):

		print method

		Y = manifold.LocallyLinearEmbedding(n_components=n_components, 
			n_neighbors=n_neighbors, method=method, eigen_solver='auto').fit_transform(X)
		ax = plt.subplot(241+i)
		plt.scatter(Y[:,0], Y[:,1])
		plt.title("LLE-%s" % method)

	Y = manifold.Isomap(n_components=n_components).fit_transform(X)
	ax = plt.subplot(255)
	plt.scatter(Y[:,0], Y[:,1])
	plt.title("Isomap")

	Y = manifold.MDS(n_components=n_components).fit_transform(X)
	ax = plt.subplot(256)
	plt.scatter(Y[:,0], Y[:,1])
	plt.title("MDS")
	
	Y = manifold.SpectralEmbedding(n_components=n_components).fit_transform(X)
	ax = plt.subplot(257)
	plt.scatter(Y[:,0], Y[:,1])
	plt.title("Spectral")
	
	Y = manifold.TSNE(n_components=n_components).fit_transform(X)
	ax = plt.subplot(258)
	plt.scatter(Y[:,0], Y[:,1])
	plt.title("TSNE")

	plt.savefig(outpath)
	