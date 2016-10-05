import rpy2.robjects as ro
import rpy2.robjects.numpy2ri
rpy2.robjects.numpy2ri.activate()
from rpy2.robjects.packages import importr
import default_params as depa
from running_utils import to_cov
import numpy as np
eps = 1e-6

def clime(data, alpha = depa.default_alpha):
	C = to_cov(data)
	d = {'fastclime.lambda': 'fastclime_lamb'}
	fastclime = importr('fastclime', robject_translations = d)
	out1 = fastclime.fastclime(C)
	O = fastclime.fastclime_lamb(out1[4], out1[5], alpha)
	return np.array(O[0])

def custom_clime(data, TPC):
	C = to_cov(data)
	d = {'fastclime.lambda': 'fastclime_lamb'}
	fastclime = importr('fastclime', robject_translations = d)
	out1 = fastclime.fastclime(C)
	yla = depa.default_alpha
	P = np.array(fastclime.fastclime_lamb(out1[4], out1[5], yla)[0])
	while (sum(sum(1 for i in l if abs(i) > 0) for l in P) > TPC):
		yla *= 2
		if (yla*eps > 1):
			break
		np.array(fastclime.fastclime_lamb(out1[4], out1[5], yla)[0])
	ala = yla/2
	P = np.array(fastclime.fastclime_lamb(out1[4], out1[5], ala)[0])
	while (sum(sum(1 for i in l if abs(i) > 0) for l in P) < TPC):
		ala /= 2
		if ala < eps:
			break
		P = np.array(fastclime.fastclime_lamb(out1[4], out1[5], ala)[0])
	for _ in xrange(8):
		kes = (ala + yla)/2
		P = np.array(fastclime.fastclime_lamb(out1[4], out1[5], kes)[0])
		TC = sum(sum(1 for i in l if abs(i) > 0) for l in P)
		if (TC == TPC):
			break
		if (TC > TPC):
			ala = kes
		else:
			yla = kes
	P = np.array(fastclime.fastclime_lamb(out1[4], out1[5], kes)[0])
	return P

def clime_precision(data, TPC):
	C = to_cov(data)
	d = {'fastclime.lambda': 'fastclime_lamb'}
	fastclime = importr('fastclime', robject_translations = d)
	out1 = fastclime.fastclime(C)
	yla = depa.default_alpha
	P = np.array(fastclime.fastclime_lamb(out1[4], out1[5], yla)[0])
	while (sum(sum(1 for i in l if abs(i) > 0) for l in P) > TPC):
		yla *= 2
		if (yla*eps > 1):
			break
		np.array(fastclime.fastclime_lamb(out1[4], out1[5], yla)[0])
	ala = yla/2
	P = np.array(fastclime.fastclime_lamb(out1[4], out1[5], ala)[0])
	while (sum(sum(1 for i in l if abs(i) > 0) for l in P) < TPC):
		ala /= 2
		if ala < eps:
			break
		P = np.array(fastclime.fastclime_lamb(out1[4], out1[5], ala)[0])
	for _ in xrange(8):
		kes = (ala + yla)/2
		P = np.array(fastclime.fastclime_lamb(out1[4], out1[5], kes)[0])
		TC = sum(sum(1 for i in l if abs(i) > 0) for l in P)
		if (TC == TPC):
			break
		if (TC > TPC):
			ala = kes
		else:
			yla = kes
	P = np.array(fastclime.fastclime_lamb(out1[4], out1[5], kes)[0])
	return P