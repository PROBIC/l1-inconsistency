import numpy as np
import default_params as depa
from running_utils import get_P

def generate_seed_data(data_params, seed, coeff = 1.0):
	np.random.seed(seed)
	data_graph_pairs = []
	for data_param in data_params:
		indim, outdim, samples, noise_std = data_param
		raw_data = generate_case(indim, outdim, samples, noise_std, only_shape=False, coeff = coeff)
		Graph = generate_case(indim, outdim, samples, noise_std, only_shape=True)
		data_graph_pairs.append((raw_data, Graph))
	return data_graph_pairs

def generate_seed_data_L2(data_params, seed, coeff = 1.0):
	np.random.seed(seed)
	data_P_pairs = []
	for data_param in data_params:
		indim, outdim, samples, noise_std = data_param
		raw_data, real_P = generate_case_P(indim, outdim, samples, noise_std, coeff = coeff)
		Graph = generate_case(indim, outdim, samples, noise_std, only_shape=True)
		data_P_pairs.append((raw_data, real_P, Graph))
	return data_P_pairs

def generate_seed_As(data_params, seed):
	np.random.seed(seed)
	As = []
	for data_param in data_params:
		indim, outdim, samples, noise_std = data_param
		A = generate_A(indim, outdim, samples, noise_std, only_shape=False)
		Graph = generate_case(indim, outdim, samples, noise_std, only_shape=True)
		As.append(A)
	return As

def generate_case(indim = depa.default_indim, outdim = depa.default_outdim, samples = depa.default_sample_size, noise_std=depa.default_noise_std, only_shape=False, coeff = 1.0):
	if (only_shape):
		return np.array([[1 if ((min(i,j) < indim) or (i==j)) else 0 for i in xrange(indim + outdim)] for j in xrange(indim + outdim)])
	A = np.random.randn(indim, samples)
	X = np.random.randn(outdim, indim)*coeff
	B = np.dot(X, A) + noise_std*np.random.randn(outdim, samples)
	B = B / np.std(B, 1)[:,np.newaxis]
	return np.vstack((A, B))

def generate_case_P(indim = depa.default_indim, outdim = depa.default_outdim, samples = depa.default_sample_size, noise_std=depa.default_noise_std, coeff = 1.0):
	A = np.random.randn(indim, samples)
	X = np.random.randn(outdim, indim)*coeff
	B = np.dot(X, A) + noise_std*np.random.randn(outdim, samples)
	B = B / np.std(B, 1)[:,np.newaxis]
	return np.vstack((A, B)), get_P(X, noise_std)

def generate_A(indim = depa.default_indim, outdim = depa.default_outdim, samples = depa.default_sample_size, noise_std=depa.default_noise_std, only_shape=False):
	if (only_shape):
		return np.array([[1 if ((min(i,j) < indim) or (i==j)) else 0 for i in xrange(indim + outdim)] for j in xrange(indim + outdim)])
	A = np.random.randn(indim, samples)
	X = np.random.randn(outdim, indim)
	B = np.dot(X, A) + noise_std*np.random.randn(outdim, samples)
	B = B / np.std(B, 1)[:,np.newaxis]
	return X

def generate_doubled_case(indim, outdim, samples, noise_std = depa.default_noise_std, noise2 = 0.01):
	A = np.random.randn(indim, samples)
	Ap = []
	row1 = np.random.randn(samples)*noise2 + A[0]
	row2 = A[0]
	Ap = [row1, row2]
	for i in xrange(1, indim):
		Ap.append(A[i])
	Ap = np.array(Ap)
	X = np.random.randn(outdim, indim)
	B = np.dot(X, A) + noise_std*np.random.randn(outdim, samples)
	B = B / np.std(B, 1)[:,np.newaxis]
	return np.vstack((A, B)), np.vstack((Ap, B))