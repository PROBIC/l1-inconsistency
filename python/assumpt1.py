# 	Testing for assumption one
# 
# 	assumption1_norm(C, Graph) takes as the imput
# 		- The covariance matrix
# 		- The Graph (in the form of a matrix with 1 at the edges and the diagonal, and 0 otherwise)
# 	and outputs the l_infty-norm of the matrix in the assumption 1. 

from __future__ import print_function
import numpy as np


def p_print(X):
	for row in X:
		for i in row:
			print (str(int(round(i))) + " ", end="")
		print()


def case_graph(indim, outdim):
	A11 = np.ones((indim, indim))
	A12 = np.ones((indim, outdim))
	A21 = np.ones((outdim, indim))
	A22 = np.eye((outdim))
	return np.array(np.bmat([[A11, A12],[A21, A22]]))

def random_case_covariance(indim, outdim, noise_std):
	A = np.random.randn(outdim, indim)
	A11 = np.eye(indim)
	A12 = np.zeros((indim,outdim))
	A21 = A
	A22 = noise_std*np.eye(outdim)
	M = np.array(np.bmat([[A11, A12], [A21, A22]]))
	return np.dot(M, np.transpose(M))

def hessian(C):
	p_print(np.kron(C, C))
	return np.kron(C, C)

def slice_indeces(Graph, complement=False):
	dim = Graph.shape[1]
	return [i + dim*j for i in range(dim) for j in range(dim) if ((Graph[i][j]==0) == complement)]

def gamma_slice(H, Graph, complement=False):
	sl1 = slice_indeces(Graph, complement=complement)
	sl2 = slice_indeces(Graph, complement=False)
	return H[np.ix_(sl1, sl2)]

def assumption1_norm(C, Graph):
	H = hessian(C)
	G1 = gamma_slice(H, Graph, complement=True)
	G2 = gamma_slice(H, Graph, complement=False)
	M = np.dot(G1, np.linalg.inv(G2))
	return np.linalg.norm(M, ord=np.inf)

def assumption1_matrix(C, Graph):
	H = hessian(C)
	G1 = gamma_slice(H, Graph, complement=True)
	G2 = gamma_slice(H, Graph, complement=False)
	M = np.dot(G1, np.linalg.inv(G2))
	II = np.linalg.inv(G2)
	return M

def tester(indim, outdim, noise_std):
	Graph = case_graph(indim, outdim)
	while (True):
		C = random_case_covariance(indim, outdim, noise_std)
		print(assumption1_norm(C, Graph))

indim, outdim = 2, 10
noise_std = 0.1

#tester(indim, outdim,noise_std)
