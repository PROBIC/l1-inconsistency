# contains the code for performing the gene expression structure learning experiment for glasso 
import numpy as np
import numpy.random as npr
import numpy.linalg as lg
import pandas as pd
import os
import matplotlib.pyplot as plt
from running_utils import *
import sys
import pickle
from sklearn.preprocessing import scale

# load data, path needs to be changed
def load_data(path = "~/Documents/datas/tcga_brca/genomicMatrix"):
    dt = pd.read_table(os.path.expanduser(path), index_col=0)
    dt = dt.loc[dt.std(1) != 0,]
    return dt.values
 
 # make matrix symmetric   
def symmetric(A):
	return (A + A.T)/2

# create new covariance, precision and adjacency matrices from given data by thresholding the observed precision
def createCov(dataMat, dim, CUTOFF = 0.1):
	done = False
	# check that created cov matrix will be positive definite
	while not done:
		J = npr.choice(dataMat.shape[0], dim, False)
		obscov = np.corrcoef(dataMat[J,])
		obsprec = lg.inv(obscov)
		newprec = obsprec
        
		newprec[abs(newprec) < CUTOFF] = 0
        
		newcov = lg.inv(newprec)
	
		newcov = symmetric(newcov) # make matrices symmetric (there seemed to be little differences between elements, like ~1e-17, but still..)
		newprec = symmetric(newprec)
        
		if isPosDef(newcov):
			done = True
            
	return newcov, newprec
	
# check that matrix is pos. definite	
def isPosDef(cov, thres = 1e-12):
	return np.all(np.linalg.eigvals(cov) > thres)
		
# given covariance matrix, sample zero mean multivariate normal data		
def sampleMVnorm(cov,samples):
	if isPosDef(cov) == False:
		print "Input covariance is not positive-definite"
		sys.exit()
	
	d = cov.shape[0]	
	X = npr.multivariate_normal(np.zeros(d), cov,samples) # sample zero mean mv-normal data
	
	return X
	
def saveRes(res,filename):
	pickle.dump(res,open(filename,"wb"))
		
def loadRes(filename):
	return pickle.load(open(filename,"rb"))
	
# Function to run the actual tests	
def glassoTests(dims,samples,nTests,seed,methods):
	npr.seed(seed)
	
	D = load_data()
		
	maxSample = np.amax(samples)
	
	# initialize matrices to collect results
	glassoHD = np.zeros((len(samples),len(dims),nTests))
	randomGuess = np.zeros((2,len(dims),nTests))
	glassoPREC = np.zeros((len(samples),len(dims),nTests))
	glassoREC = np.zeros((len(samples),len(dims),nTests))
	convGfails = np.zeros((len(samples),len(dims)))

	for ttt in range(nTests):
		jj = 0	
		for dim in dims:
			mats = createCov(D, dim,0.1) #cutoff 0.1 
			trueCov = mats[0]
			truePrec = mats[1]
			trueGraph = 1.0*(abs(truePrec) > 0)
			
			nEdges = sum(sum(trueGraph - np.diag(np.diag(trueGraph))))/2 # number of edges in the true graph
			totalEDGE = dim*(dim-1)/2 # number of possible edges
			NONedges =  totalEDGE - nEdges # number of non-edges
			
			X = sampleMVnorm(trueCov,maxSample)
				
			Cs = []
			for n in samples: # compute sample covariances for different sample sizes
				Xn = X[0:n,:]
				print Xn.shape
				Xn = scale(Xn) # center and scale
				print np.cov(Xn.T).shape
				Cs.append(np.cov(Xn.T))
					
			res = evaluate_methods(Cs,trueGraph,methods)
			pre_recs = res["glasso"]
			
			# store results for every sample size
			for i in range(len(samples)):
				
				glassoHD[i,jj,ttt] = glassoHD[i,jj,ttt] + hamming_distance(pre_recs[i][0], pre_recs[i][1], nEdges)
				
				# glasso did not converge, result is nan
				if(pre_recs[i][0] == 0 and pre_recs[i][1] == 0 ):
						convGfails[i,jj] += 1
						glassoPREC[i,jj,ttt] = np.nan
						glassoREC[i,jj,ttt] = np.nan
				else:				
					glassoPREC[i,jj,ttt] = pre_recs[i][0]
					glassoREC[i,jj,ttt] = pre_recs[i][1]
					
			#results for random guessing
			randHD = 2.0*(nEdges*NONedges)/(nEdges+NONedges) # hamming distance by random guessing
			randPREC = 1.0*nEdges/totalEDGE # precision by random guessing (edge density)
			
			randomGuess[0, jj, ttt] = randHD
			randomGuess[1, jj, ttt] = randPREC
			
			
			jj = jj + 1
			
	#meanHD = np.nanmean(glassoHD,2) 	
			
	res = {"HDs" : glassoHD, "dims" : dims, "samples" : samples, "nTests" : nTests, "seed" : seed,  "precs" : glassoPREC, "recalls" : glassoREC, "converg" : convGfails, "guess": randomGuess}			
				
	return res
		
def hamming_distance(precision, recall, nEdges):
	TP = (2*nEdges*recall)
	FP = (TP/precision - TP)
	hamming_distance = 2*nEdges - TP + FP
	print nEdges, precision, recall, hamming_distance
	return hamming_distance/2
		
# For the results used in the paper: "printGlasso("../genExpResults/glasso0703.p"); plt.show()" 
def printGlasso(filename, yAxis = "precision", randomGuess = True):
	
	widthhh = 2.5
	fig = plt.figure()
	plt.xlabel("Dimension")
	
	if(yAxis == "HD"):
		plt.ylabel("Hamming distance")
	elif(yAxis == "precision"):
		plt.ylabel("Precision")
		
				
	glassoRes = loadRes(filename)
	
	HDs = glassoRes["HDs"]
	HDs = np.nanmean(HDs,2)
	
	precs = glassoRes["precs"]
	precs = np.nanmean(precs,2) 
	
	dims = glassoRes["dims"]
	samples = glassoRes["samples"]
	
	x = range(len(dims))
	
	for sample in samples:
		row = samples.index(sample)
		label = "$n$ = " + str(sample)
		
		if(yAxis == "HD"):
			y = HDs[row,:]
		elif(yAxis == "precision"):
			y = precs[row,:]
		
		plt.plot(x,y,label = label, linewidth = widthhh)
		
	if(randomGuess):
		Gavgs = glassoRes["guess"]
		Gavgs = np.mean(Gavgs,2)

		if(yAxis == "HD"):
			y = Gavgs[0,:]
		elif(yAxis == "precision"):
			y = Gavgs[1,:]
		
		plt.plot(x,y,label = "Random",linewidth = widthhh)
	
	
	plt.xticks(x, dims)
	
	if(yAxis == "HD"):
		plt.legend(loc = 2)
	elif(yAxis == "precision"):
		plt.ylim(0,1)
		plt.legend(loc = 3)
		
		
	plt.rcParams.update({'font.size': 18})

	return fig
