from running_utils import *
from saving_utils import *
from data_generation import *
import default_params as depa
from case_math_test import custom_specs, custom_try, custom_test
import matplotlib.pyplot as plt
from glasso_objective import *
from matplotlib import rc
rc('text', usetex=True)

def test_glasso_A1_norm(K = 10, indim=2, outdim=10, noise_std=0.1):
	# Values used in the paper: K = 200, indim = 2, outdim = 10, noise_std = 0.1, seed = 1234
	xs = []
	ys = []
	Cs = []
	np.random.seed(1234)
	for i in xrange(K):
		A = np.random.randn()*np.random.randn(outdim, indim)
		C, Graph = custom_specs(A, noise_std)
		a1 = custom_try(A, noise_std)
		a2 = custom_test(A, noise_std)
		assert((a1- a2) < 1e-6)
		xs.append(a1)
		Cs.append(C)
	data = evaluate_methods(Cs, Graph, ["glasso"])["glasso"]
	ys = [yy[0] for yy in data] 
	fig = plt.figure()
	plt.plot(xs, ys, '*', label="glasso")
	plt.plot([1,1], [0,1.1], linewidth=5)
	axes = plt.gca()
	axes.set_ylim([0, 1.1])
	fontsize = 20
	axes.set_ylabel("precision", fontsize=fontsize)
	axes.set_xlabel(r"$\gamma$", fontsize=fontsize)
	axes.set_xscale('log')
	return fig

def test_A1_norm(K = 10, indim=2, outdim=10, noise_std=0.1, methods = ["glasso",]):
	xs = []
	ys = []
	Cs = []
	np.random.seed(1234)
	for i in xrange(K):
		A = np.random.randn()*np.random.randn(outdim, indim)
		C, Graph = custom_specs(A, noise_std)
		a1 = custom_try(A, noise_std)
		xs.append(a1)
		Cs.append(C)
	data = evaluate_methods(Cs, Graph, methods)
	print methods
	plot_data = {method : zip(*sorted(zip(xs, data[method]))) for method in methods}
	fig = plt.figure()
	for method in plot_data:
		print method
		plt.plot(np.log(plot_data[method][0]), plot_data[method][1],'*', label=method)
	plt.legend()
	axes = plt.gca()
	axes.set_ylim([0, 1.1])
	plt.plot([0,0], [0,1.1], linewidth=5)
	return fig