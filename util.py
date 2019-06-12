from numpy import array, zeros, float32, log2, flatnonzero
import time
from line_profiler import LineProfiler
import igraph as ig



def timeit(fun):
	def timed(*argst, **kwt):
		ts = time.time()
		result = fun(*argst, **kwt)
		te = time.time()
		print(fun.__name__, te - ts)
		return result
	return timed
	

def lp_wrapper(f):
	lp = LineProfiler()
	g = lp(f)
	def wrapped(*argst, **kwt):
		result = g(*argst, **kwt)
		lp.print_stats()
		return result
	return wrapped


# @lp_wrapper
def entropy(count_arr):
	count_arr = count_arr[flatnonzero(count_arr)]
	if len(count_arr) <= 1:
		return 0.0
	else:
		p = count_arr / sum(count_arr)
		ent = - p * log2(p)
		return sum(ent)


def my_modularity(G, nd_clst_idx, K):
	import numpy as np

	nd_clst_idx = np.array(nd_clst_idx)
	A = np.array(G.get_adjacency().data)
	M = np.sum(A)
	E = np.zeros((K,K))

	for i in range(K):
		E[i,i] = np.sum(A[nd_clst_idx==i,:][:,nd_clst_idx==i]) / M

	for i in range(0, K-1):
		for j in range(i+1, K):
			E[i,j] = np.sum(A[nd_clst_idx==i,:][:,nd_clst_idx==j]) / M
			E[j,i] = E[i,j]

	Alpha = E.sum(axis=1)
	modularity = np.sum(np.diag(E) - np.square(Alpha))
	return modularity

def igraph_modularity(G, nd_clst_idx):
	return G.modularity(nd_clst_idx)

def igraph_nmi(l1, l2):
	return ig.compare_communities(l1, l2, "nmi", remove_none=True)

def cluster_density(G, cluster, M):
	edge_count = 0
	for clu in cluster:
		InduSubG = G.induced_subgraph(clu)
		edge_count += len(InduSubG.es)
	den = float(edge_count) / M
	return den


def cluster_enrtopy(clst_attr_cnt, K, N, W, cluster):
	ent_sum = 0.0
	attr_dim = len(clst_attr_cnt)
	W_sum = sum([W[i][0] for i in range(attr_dim)])
	for i in range(K):
		ent_i = 0.0
		for j in range(attr_dim):
			ent_i_j = entropy(array(list(clst_attr_cnt[j][i].values())))
			ent_i += (W[j][i] / W_sum) * ent_i_j
		ent_sum += (float(len(cluster[i])) / N) * ent_i
	return ent_sum


def attr_enrtopy(clst_attr_cnt, K, N, W, cluster):
	attr_dim = len(clst_attr_cnt)
	ent_attr = []
	for j in range(attr_dim):
		ent_j = 0.0
		for i in range(K):
			ent_i_j = entropy(array(list(clst_attr_cnt[j][i].values())))
			ent_j += (float(len(cluster[i])) / N) * ent_i_j
		ent_attr.append(ent_j)
	return ent_attr
