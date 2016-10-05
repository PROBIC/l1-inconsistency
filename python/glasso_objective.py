import numpy as np
import numpy.linalg as npl

from running_utils import optimise_param
from glasso import glasso
import sklearn.covariance.graph_lasso_
from sklearn.utils.extmath import fast_logdet


def glasso_objective_(myprec, mycov, alpha=1.0):
    v = np.array((fast_logdet(myprec), -np.sum(myprec * mycov),
                  -alpha*(np.abs(myprec).sum() - np.abs(np.diag(myprec)).sum())))
    return np.append(v, [np.sum(v), sklearn.covariance.graph_lasso_._objective(mycov, myprec, alpha), alpha])

def glasso_objective_split(newprec, newcov):
    TPC = np.sum(newprec != 0)
    glassoprec, alpha = optimise_param(glasso, newcov, TPC, return_param=True)
    return (glasso_objective_(newprec, newcov, alpha), glasso_objective_(glassoprec, newcov, alpha))
