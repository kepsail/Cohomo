from random import shuffle

from load_data import Data
from util import *


class Cohomo():

	@timeit
	def __init__(self, data):
		self.G = data.UG
		self.A = data.A
		self.attr_dim = data.attr_dim

		VertexClustering = data.VertexClustering

		self.N = len(self.G.vs)
		self.M = len(self.G.es)
		self.K = len(VertexClustering)
		
		cluster = []
		nd_clst_idx = [-1] * self.N
		clst_attr_cnt = []
		for j in range(self.attr_dim):
			clst_attr_cnt.append([])

		for i in range(self.K):
			cluster.append(VertexClustering[i])

			for j in range(self.attr_dim):
				clst_attr_cnt[j].append({})

			for node in cluster[i]:
				nd_clst_idx[node] = i

				for j in range(self.attr_dim):
					clst_attr_cnt[j][i][self.A[j][node]] = clst_attr_cnt[j][i].setdefault(self.A[j][node], 0) + 1
		
		self.cluster = cluster
		self.nd_clst_idx = nd_clst_idx
		self.clst_attr_cnt = clst_attr_cnt


	def init_attr_weight(self):
		W = []
		init_W = float(1) / self.attr_dim * (self.attr_dim -1)
		for j in range(self.attr_dim):
			W.append([init_W]*self.K)
		self.W = W


	@timeit
	@lp_wrapper
	def update_attr_weight(self):
		for i in range(self.K):
			ent = []
			for j in range(self.attr_dim):
				ent.append(entropy(array(list(self.clst_attr_cnt[j][i].values()))))

			ent_sum = sum(ent)
			if ent_sum == 0.0:
				for j in range(self.attr_dim):
					self.W[j][i] = float(1) / self.attr_dim * (self.attr_dim -1)
			else:
				for j in range(self.attr_dim):
					self.W[j][i] = 1 - ent[j] / ent_sum


	@timeit
	@lp_wrapper
	def adjust_sync(self, sample_interval=1, max_iter=15):

		G = self.G
		A = self.A
		cluster = self.cluster
		clst_attr_cnt = self.clst_attr_cnt
		nd_clst_idx = self.nd_clst_idx

	
		iter_time = 0
		while iter_time < max_iter:
			iter_time += 1
			adj_num = 0
			for x in range(self.K):
				for node in cluster[x]:
					for nb in G.neighbors(node):
						if nd_clst_idx[nb] != x:
							
							y = nd_clst_idx[nb]
							n_x = len(cluster[x])
							n_y = len(cluster[y])

							ent_x_w_bef = ent_x_w_aft = ent_y_w_bef = ent_y_w_aft = 0.0

							for j in range(self.attr_dim):
								clst_x_attr_cnt = clst_attr_cnt[j][x].copy()
								clst_y_attr_cnt = clst_attr_cnt[j][y].copy()
								
								ent_x_w_bef += self.W[j][x] * entropy(array(list(clst_x_attr_cnt.values())))
								ent_y_w_bef += self.W[j][y] * entropy(array(list(clst_y_attr_cnt.values())))

								clst_x_attr_cnt[A[j][nb]] = clst_x_attr_cnt.setdefault(A[j][nb], 0) + 1
								clst_y_attr_cnt[A[j][nb]] -= 1
								
								ent_x_w_aft += self.W[j][x] * entropy(array(list(clst_x_attr_cnt.values())))
								ent_y_w_aft += self.W[j][y] * entropy(array(list(clst_y_attr_cnt.values())))

							delta_obj_value = (n_x + 1) * ent_x_w_aft - n_x * ent_x_w_bef + (n_y - 1) * ent_y_w_aft - n_y * ent_y_w_bef
							if delta_obj_value < 0.0:

								adj_num += 1

								cluster[y].remove(nb)
								cluster[x].append(nb)
								nd_clst_idx[nb] = x

								for j in range(self.attr_dim):
									clst_attr_cnt[j][x][A[j][nb]] = clst_attr_cnt[j][x].setdefault(A[j][nb], 0) + 1
									clst_attr_cnt[j][y][A[j][nb]] -= 1

			print('adj_num: ', adj_num, '\n')

			self.update_attr_weight()
			if adj_num == 0:
				break	

	@timeit
	@lp_wrapper
	def adjust_async(self, sample_interval=1, max_iter=15):

		G = self.G
		A = self.A
		cluster = self.cluster
		clst_attr_cnt = self.clst_attr_cnt
		nd_clst_idx = self.nd_clst_idx


		iter_time = 0
		while iter_time < max_iter:
			iter_time += 1
			adj_num = 0
			for x in range(self.K):
				for node in cluster[x]:
					for nb in G.neighbors(node):
						if nd_clst_idx[nb] != x:
							
							y = nd_clst_idx[nb]
							n_x = len(cluster[x])
							n_y = len(cluster[y])
						
							ent_x_aft = []
							ent_y_aft = []
							
							ent_x_w_bef = ent_x_w_aft = ent_y_w_bef = ent_y_w_aft = 0.0

							for j in range(self.attr_dim):
								clst_x_attr_cnt = clst_attr_cnt[j][x].copy()
								clst_y_attr_cnt = clst_attr_cnt[j][y].copy()
								
								ent_x_w_bef += self.W[j][x] * entropy(array(list(clst_x_attr_cnt.values())))
								ent_y_w_bef += self.W[j][y] * entropy(array(list(clst_y_attr_cnt.values())))

								clst_x_attr_cnt[A[j][nb]] = clst_x_attr_cnt.setdefault(A[j][nb], 0) + 1
								clst_y_attr_cnt[A[j][nb]] -= 1
								
								ent_x_aft.append(entropy(array(list(clst_x_attr_cnt.values()))))
								ent_y_aft.append(entropy(array(list(clst_y_attr_cnt.values()))))

								ent_x_w_aft += self.W[j][x] * ent_x_aft[j]
								ent_y_w_aft += self.W[j][y] * ent_y_aft[j]

							delta_obj_value = (n_x + 1) * ent_x_w_aft - n_x * ent_x_w_bef + (n_y - 1) * ent_y_w_aft - n_y * ent_y_w_bef
							if delta_obj_value < 0.0:
								
								adj_num += 1
								
								cluster[y].remove(nb)
								cluster[x].append(nb)
								nd_clst_idx[nb] = x
								for j in range(self.attr_dim):
									clst_attr_cnt[j][x][A[j][nb]] = clst_attr_cnt[j][x].setdefault(A[j][nb], 0) + 1
									clst_attr_cnt[j][y][A[j][nb]] -= 1

								sum_ent_x_aft = sum(ent_x_aft)
								sum_ent_y_aft = sum(ent_y_aft)

								if sum_ent_x_aft == 0.0:
									for j in range(self.attr_dim):
										self.W[j][x] = float(1) / self.attr_dim * (self.attr_dim -1)
								else:
									for j in range(self.attr_dim):
										self.W[j][x] = 1 - ent_x_aft[j] / sum_ent_x_aft


								if sum_ent_y_aft == 0.0:
									for j in range(self.attr_dim):
										self.W[j][y] = float(1) / self.attr_dim * (self.attr_dim -1)
								else:
									for j in range(self.attr_dim):
										self.W[j][y] = 1 - ent_y_aft[j] / sum_ent_y_aft


			print('adj_num: ', adj_num, '\n')
			if adj_num == 0:
				break


if __name__ == '__main__':

	ts = time.time()

	edges_file = './datasets/polblogs/edgelist_py.txt'
	pkl_file = './datasets/polblogs/VertexClustering.pkl'
	attr1_file = './datasets/polblogs/leaning.txt'
	attr2_file = './datasets/polblogs/gender.txt'
	attr3_file = './datasets/polblogs/source.txt'

	# edges_file = './datasets/dblp10000/edgelist_py.txt'
	# pkl_file = './datasets/dblp10000/VertexClustering.pkl'
	# attr1_file = './datasets/dblp10000/gender.txt'
	# # attr2_file = './datasets/dblp10000/prolific.txt'
	# attr3_file = './datasets/dblp10000/topic.txt'

	# edges_file = './datasets/dblp84170/edgelist_py.txt'
	# pkl_file = './datasets/dblp84170/VertexClustering.pkl'
	# attr1_file = './datasets/dblp84170/prolific.txt'
	# attr2_file = './datasets/dblp84170/topic.txt'
	
	# edges_file = './datasets/AmazonLarge/edgelist_py.txt'
	# pkl_file = './datasets/AmazonLarge/VertexClustering.pkl'
	# attr1_file = './datasets/AmazonLarge/avg_rating.txt'
	# attr2_file = './datasets/AmazonLarge/sales_rank.txt'

	data = Data()
	data.read_graph(edges_file)
	
	data.load_clusters(pkl_file)
	# data.detect_clusters(stru_method='lpa')
	# data.detect_clusters(stru_method='infomap')
	
	data.read_attr(attr1_file)
	data.read_attr(attr2_file)
	data.read_attr(attr3_file)


	coho = Cohomo(data)
	del data
	coho.init_attr_weight()
	coho.update_attr_weight()		

	# coho.adjust_sync()
	coho.adjust_async()

	te = time.time()

	print('total time: ', te-ts, '\n')


