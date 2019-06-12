import igraph as ig
import pickle
import sys

from util import timeit

class Data():
	def __init__(self):
		self.A = []
		self.attr_dim = 0

	def read_graph(self, edges_file):
		try:
			DG = ig.Graph.Read(edges_file, format='edgelist')
			UG = DG.as_undirected()
		except Exception as e:
			print('Aborted! Failed to read graph'+', no such file '+str(edges_file))
			sys.exit(1)
		else:
			self.UG = UG
			print('Succeed to read graph')

	# @timeit
	def detect_clusters(self, stru_method='infomap'):
		try:
			if stru_method == 'infomap':
				VertexClustering = self.UG.community_infomap()
			elif stru_method =='lpa':
				VertexClustering = self.UG.community_label_propagation()
			else:
				print('Aborted! Please input arg of infomap or lpa')
				sys.exit(1)
		except Exception as e:
			print('Aborted! Failed to detect clusters')
			sys.exit(1)
		else:
			self.VertexClustering = VertexClustering
			print('Succeed to detect clusters')

	# @timeit
	def load_clusters(self, pkl_file):
		try:
			with open(pkl_file, 'rb') as in_file:
				VertexClustering = pickle.load(in_file)
		except Exception as e:
			print('Aborted! Failed to load clusters'+', no such file '+str(pkl_file))
			sys.exit(1)
		else:
			self.VertexClustering = VertexClustering
			print('Succeed to load clusters')


	def read_attr(self, attr_file):
		try:
			with open(attr_file) as in_file:
				attr = [int(x.strip()) for x in in_file.readlines()]
		except Exception as e:
			self.attr_dim += 1
			print('Aborted! Failed to read attr '+str(self.attr_dim)+', no such file '+str(attr_file))
			sys.exit(1)
		else:
			self.A.append(attr)
			self.attr_dim += 1
			print('Succeed to read attr '+str(self.attr_dim))