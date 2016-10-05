import numpy as np
import default_params as depa
from running_utils import to_cov
import sklearn.covariance
import sklearn.linear_model

def glasso(data, alpha = depa.default_alpha):
	C = to_cov(data)
	try:
		return sklearn.covariance.graph_lasso(C, alpha)[1]
	except:
		dim = C.shape[0]
		return np.zeros((dim, dim))

def glasso_precision(data, alpha = depa.default_alpha):
	C = to_cov(data)
	try:
		return sklearn.covariance.graph_lasso(C, alpha)[1]
	except:
		dim = C.shape[0]
		return np.zeros((dim, dim))