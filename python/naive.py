import default_params as depa
from running_utils import to_cov
import scipy.linalg

def naive(data, alpha = depa.default_alpha):
	C = to_cov(data)
	P = scipy.linalg.inv(C)
	P[abs(P) < alpha] = 0
	return P

def naive_precision(data, alpha = depa.default_alpha):
	C = to_cov(data)
	return scipy.linalg.inv(C)
