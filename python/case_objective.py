from running_utils import *
from saving_utils import *
from data_generation import *
import pickle
import os
import default_params as depa
import matplotlib.pyplot as plt
from glasso_objective import *
from matplotlib import rc
rc('text', usetex=True)

def generate_case_example(indim, outdim, sample_size, noise_std, coeff):
	A = np.random.rand(outdim, indim)*coeff
	C = get_C(A, noise_std)
	P = get_P(A, noise_std)
	return P, C

def glasso_objective(indim = depa.default_indim, outdim = depa.default_outdim, sample_size = depa.default_sample_size, \
		repeats=depa.default_repeat_count, noise_stds = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0], seed=depa.default_seed, coeff = 1.0):
	glassosplits = np.zeros((len(noise_stds), 6))
	truesplits = np.zeros((len(noise_stds), 6))
	np.random.seed(seed)
	for j, noise_std in enumerate(noise_stds):
		gsplits = np.zeros((repeats, 6))
		tsplits = np.zeros((repeats, 6))
		for i in range(repeats):
			newprec, newcov = generate_case_example(indim, outdim, sample_size, noise_std, coeff)
			while fast_logdet(newprec) < -1e10:
				newprec, newcov = generate_case_example(indim, outdim, sample_size, noise_std, coeff)
			ts, gs = glasso_objective_split(newprec, newcov)
			while gs[0] < -1e10:
				newprec, newcov = generate_case_example(indim, outdim, sample_size, noise_std, coeff)
				while fast_logdet(newprec) < -1e10:
					newprec, newcov = generate_example(indim, outdim, sample_size, noise_std, coeff)
				ts, gs = glasso_objective_split(newprec, newcov)
			tsplits[i,:] = ts
			gsplits[i,:] = gs
		glassosplits[j,:] = np.mean(gsplits, 0)
		truesplits[j,:] = np.mean(tsplits, 0)
	return (glassosplits, truesplits)

def glasso_objective_coeffs(indim = depa.default_indim, outdim = depa.default_outdim, sample_size = depa.default_sample_size, \
		repeats=depa.default_repeat_count, noise_std = 0.1, seed=depa.default_seed, coeffs = [0.01, 0.02, 0.05, 0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0, 20.0, 50.0, 100.0]):
	glassosplits = np.zeros((len(coeffs), 6))
	truesplits = np.zeros((len(coeffs), 6))
	np.random.seed(seed)
	for j, coeff in enumerate(coeffs):
		gsplits = np.zeros((repeats, 6))
		tsplits = np.zeros((repeats, 6))
		for i in range(repeats):
			newprec, newcov = generate_case_example(indim, outdim, sample_size, noise_std, coeff)
			while fast_logdet(newprec) < -1e10:
				newprec, newcov = generate_case_example(indim, outdim, sample_size, noise_std, coeff)
			ts, gs = glasso_objective_split(newprec, newcov)
			while gs[0] < -1e10:
				newprec, newcov = generate_case_example(indim, outdim, sample_size, noise_std, coeff)
				while fast_logdet(newprec) < -1e10:
					newprec, newcov = generate_example(indim, outdim, sample_size, noise_std, coeff)
				ts, gs = glasso_objective_split(newprec, newcov)
			tsplits[i,:] = ts
			gsplits[i,:] = gs
		glassosplits[j,:] = np.mean(gsplits, 0)
		truesplits[j,:] = np.mean(tsplits, 0)
	return (glassosplits, truesplits)

def plot_splits(gsplits, tsplits, noises):
	fig = plt.figure(figsize=(87.0/25.4, 70/25.4))
	plt.rcParams.update({'font.size': 6.0})
	t = range(len(noises))
	plt.plot(t, (gsplits[:, 2]), "b-")
	plt.plot(t, (tsplits[:, 2]), "b--")
	plt.plot(t, (gsplits[:, 0] + gsplits[:, 1]), "g-")
	plt.plot(t, (tsplits[:, 0] + tsplits[:, 1]), "g--")
	plt.xticks(t, noises)
	fontsize = 7.5
	plt.xlabel(r"$\sigma_{\varepsilon}$", fontsize=fontsize)
	plt.ylabel('$C$', fontsize=fontsize)
	plt.legend(["glasso penalty", "truth penalty", "glasso logl", "truth logl"], loc="center left")
	return fig


def do_plot_splits(indim = depa.default_indim, outdim = depa.default_outdim, sample_size = depa.default_sample_size, \
		repeats=depa.default_repeat_count, noise_stds = [0.1, 0.2, 0.5, 1.0, 2.0, 5.0, 10.0], seed=depa.default_seed, coeff = 1.0):
	gsplits, tsplits = glasso_objective(indim=indim, outdim=outdim, sample_size=sample_size, repeats=repeats, noise_stds=noise_stds, seed=seed, coeff=coeff)
	return plot_splits(gsplits, tsplits, noise_stds)

def do_plot_splits_coeffs(indim = depa.default_indim, outdim = depa.default_outdim, sample_size = depa.default_sample_size, \
		repeats=depa.default_repeat_count, noise_std = depa.default_noise_std, seed=depa.default_seed, coeffs = [0.05, 0.1, 0.2, 0.5, 1.0, 2.0]):
	gsplits, tsplits = glasso_objective_coeffs(indim=indim, outdim=outdim, sample_size=sample_size, repeats=repeats, noise_std=noise_std, seed=seed, coeffs=coeffs)
	return plot_splits(gsplits, tsplits, coeffs)


