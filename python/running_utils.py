import numpy as np
import default_params as depa
eps = 1e-6

def L2_distance(P, real_P):
	return np.linalg.norm(P - real_P, ord=2)

def to_cov(data):
	if (data.shape[0] != data.shape[1]):
		return np.cov(data)
	if (max(max(x for x in row) for row in abs(data - np.transpose(data))) > eps):
		return np.cov(data)
	return data

def get_A(C, indim, outdim):
	return C[indim:indim + outdim, 0:indim]

def get_C(A, noise_std):
	A = np.array(A)
	outdim, indim = A.shape
	A11 = np.eye(indim)
	A12 = np.zeros((indim,outdim))
	A21 = A
	A22 = noise_std*np.eye(outdim)
	M = np.array(np.bmat([[A11, A12], [A21, A22]]))
	C = np.dot(M, np.transpose(M))
	return C

def get_P(A, noise_std):
	outdim, indim = A.shape
	c = 1/(noise_std**2)
	A11 = np.eye(indim) + np.dot(np.transpose(A), A)*c
	A12 = -c*np.transpose(A)
	A21 = -c*A
	A22 = c*np.eye(outdim)
	P = np.array(np.bmat([[A11, A12], [A21, A22]]))
	return P

def hamming_distance(precision, recall, indim, outdim):
	TPC = indim*indim + 2*indim*outdim - indim
	if (precision < 1e-6):
		A = TPC
		B = (indim + outdim)*(indim + outdim - 1) - 1 - A
		return A*B/(A + B)
	TP = round(TPC*recall)
	FP = round(TP/precision - TP)
	hamming_distance = TPC - TP + FP
	return hamming_distance/2

def evaluate_methods(Cs, Graph, methods = []):
	TPC = sum(sum(1 for x in row if x) for row in Graph)
	data = {}
	for method in methods:
		if (method == 'clime'):
			clime_module = __import__(method)
			prs = []
			for C in Cs:
				P = clime_module.custom_clime(C, TPC)
				pr = precision_recall(P, Graph)
				prs.append(pr)
			data[method] = prs
		elif (method == 'FMPL'):
			fmpl_module = __import__('FMPLpyth')
			prs1, prs2, prs3 = [], [], []
			for C in Cs:
				P1, P2, P3 = fmpl_module.fmpl(C, 0)
				pr1 = precision_recall(P1, Graph)
				pr2 = precision_recall(P2, Graph)
				pr3 = precision_recall(P3, Graph)
				prs1.append(pr1)
				prs2.append(pr2)
				prs3.append(pr3)
			data['OR'] = prs1
			data['AND'] = prs2
			data['HC'] = prs3
		else:
			method_function =  getattr(__import__(method), method)
			prs = []
			for C in Cs:
				d = optimise_param(method_function, C, TPC)
				pr = precision_recall(d, Graph)
				prs.append(pr)
			data[method] = prs
	return data

def evaluate_methods_L2(CPs, Graph, methods = []):
	TPC = sum(sum(1 for x in row if x) for row in Graph)
	data = {}
	for method in methods:
		if (method == 'clime'):
			clime_module = __import__(method)
			prs = []
			for C, real_P in CPs:
				P = clime_module.clime_precision(C, TPC)
				pr = L2_distance(P, real_P)
				prs.append(pr)
			data[method] = prs
		else:
			method_function =  getattr(__import__(method), method + "_precision")
			prs = []
			for C, real_P in CPs:
				P = optimise_param(method_function, C, TPC)
				pr = L2_distance(P, real_P)
				prs.append(pr)
			data[method] = prs
	return data

def optimise_param(method_function, raw_data, TPC, return_param=False):
	yla = depa.default_alpha
	P = method_function(raw_data, alpha=yla)
	while (sum(sum(1 for i in l if abs(i) > 0) for l in P) > TPC):
		yla *= 2
		if (yla*eps > 1):
			break
		P = method_function(raw_data, alpha=yla)
	ala = yla/2
	P = method_function(raw_data, alpha=ala)
	while (sum(sum(1 for i in l if abs(i) > 0) for l in P) < TPC):
		ala /= 2
		if ala < eps:
			break
		P = method_function(raw_data, alpha=ala)
	for _ in range(8):
		kes = (ala + yla)/2
		P = method_function(raw_data, alpha=kes)
		TC = sum(sum(1 for i in l if abs(i) > 0) for l in P)
		if (TC == TPC):
			break
		if (TC > TPC):
			ala = kes
		else:
			yla = kes
	P = method_function(raw_data, alpha=kes)
	if return_param:
		return P, kes
	else:
		return P

def precision_recall(P_est, Graph, cutoff=0.0):
	TP, FP, TPC = 0, 0, 0
	for i,row in enumerate(np.dstack((P_est, Graph))):
		for j,p in enumerate(row):
			if (i != j):
				if (p[1]):
					TPC += 1
					if (abs(p[0]) > cutoff):
						TP += 1
				elif (abs(p[0]) > cutoff):
					FP += 1
	if (TP + FP == 0):
		return 0, 0
	return 1.0*TP/(TP + FP), 1.0*TP/TPC
