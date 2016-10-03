# For the FMPL Matlab codes, please contact Janne Leppä-aho, janne.leppa-aho at cs.helsinki.fi
import numpy as np
import matlab.engine
import os

def fmpl(dataMat, prior = 0):
	X = matlab.double(dataMat.tolist()) #convert to matlab double array
	
	eng = matlab.engine.start_matlab()
	eng.cd(os.getcwd()) #
	eng.addpath('../FMPL')
	res = eng.FMPLforPY(X, prior, nargout=3)
	
	ORm = res[0]
	OR = np.array(ORm._data.tolist())
	OR = OR.reshape(ORm.size).transpose()
	np.fill_diagonal(OR,1)
	
	ANDm = res[1]
	AND = np.array(ANDm._data.tolist())
	AND = AND.reshape(ANDm.size).transpose()
	np.fill_diagonal(AND,1)
	
	HCm = res[2]
	HC = np.array(HCm._data.tolist())
	HC = HC.reshape(HCm.size).transpose()
	np.fill_diagonal(HC,1)
	
	return OR, AND, HC
	
	
