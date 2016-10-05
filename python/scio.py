import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
import default_params as depa
from running_utils import to_cov
import numpy as np

def scio(data, alpha = depa.default_alpha):
	C = to_cov(data)
	scio = importr('scio')
	return np.array(scio.scio(C, alpha)[0])

def scio_precision(data, alpha = depa.default_alpha):
	C = to_cov(data)
	scio = importr('scio')
	return np.array(scio.scio(C, alpha)[0])