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

name_dict = {"precision" : "precision", "recall" : "recall", "mini" : "minimum of precision and recall", \
			"noise_stds" : r"$\sigma_{\varepsilon}$", "hamming" : "hamming distance", "outdim" : "$d_{2}$", "indim" : "$d_{1}$"}

def make_averages(dir_name, test_type="noise_stds", val_type="hamming", methods=None):
	if (methods == None):
		methods = []
		for method in os.listdir(dir_name):
			if (method in ["glasso", "naive", "clime", "scio", "OR"]):
				methods.append(method)
	data_set = {}
	f = open(dir_name + '/' + 'params','r')
	_ , init_args = pickle.load(f)
	f.close()
	indim, outdim, sample_size, repeats, noise_std, seed = init_args
	for method in methods:
		f = open(dir_name + '/' + method, 'r')
		data = pickle.load(f)
		f.close()
		params = []
		averages = []
		for data_sets in data:
			params.append(data_sets[1])
			val = 0
			if (test_type == "outdim"):
				outdim = data_sets[1]
			if (test_type == "indim"):
				indim = data_sets[1]
			if val_type == "hamming":
				val = sum(hamming_distance(x[0], x[1], indim, outdim) for x in data_sets[0])/len(data_sets[0])
			elif val_type == "precision":
				val = sum(x[0] for x in data_sets[0])/len(data_sets[0])
			elif val_type == "recall":
				val = sum(x[1] for x in data_sets[0])/len(data_sets[0])
			elif val_type == "mini":
				val = sum(min(x[0], x[1]) for x in data_sets[0])/len(data_sets[0])
			averages.append(val)
		data_set[method] = {test_type : params, val_type : averages}
	if (test_type == "outdim"):
		f = open(dir_name + '/' + 'params','r')
		_ , init_args = pickle.load(f)
		f.close()
		indim, outdims, sample_size, repeats, noise_std, seed = init_args
		dimi = []
		for outdim in outdims:
			dimi.append(outdim*(outdim-1)*(2*indim*outdim+indim*indim-indim)/((outdim + indim - 1)*(outdim + indim)))
		data_set["random"] = {test_type : outdims, val_type : dimi}
	if (test_type == "noise_stds"):
		dimi = []
		for noise in noise_std:
			dimi.append(29.6)
		data_set["random"] = {test_type : noise_std, val_type : dimi}
	if (test_type == "indim"):
		f = open(dir_name + '/' + 'params','r')
		_ , init_args = pickle.load(f)
		f.close()
		indims, outdim, sample_size, repeats, noise_std, seed = init_args
		dimi = []
		for indim in indims:
			dimi.append(outdim*(outdim-1)*(2*indim*outdim+indim*indim-indim)/((outdim + indim - 1)*(outdim + indim)))
		data_set["random"] = {test_type : indims, val_type : dimi}
	return data_set

def plot_averages(dir_name, test_type="noise_stds", val_type="hamming", methods=None):
	data_set = make_averages(dir_name, test_type, val_type, methods)
	fig = make_figuree(data_set, test_type, val_type, name_dict[test_type], name_dict[val_type], "")
	return fig

def run_data_outdim(data_chunks, outdims, method):
	if (method != 'FMPL'):
		out = []
		for dcg, outdim in zip(data_chunks, outdims):
			res = evaluate_methods(dcg[0], dcg[1], [method])
			out.append((res[method], outdim))
		return out
	else:
		out1, out2, out3 = [], [], []
		for dcg, outdim in zip(data_chunks, outdims):
			res = evaluate_methods(dcg[0], dcg[1], [method])
			out1.append((res['OR'], outdim))
			out2.append((res['AND'], outdim))
			out3.append((res['HC'], outdim))
		return (out1, out2, out3)

def run_data_outdim_L2(data_chunks, outdims, method):
	out = []
	for dcg, outdim in zip(data_chunks, outdims):
		res = evaluate_methods_L2(dcg[0], dcg[1], [method])
		out.append((res[method], outdim))
	return out

def run_data_indim(data_chunks, indims, method):
	if (method != 'FMPL'):
		out = []
		for dcg, indim in zip(data_chunks, indims):
			res = evaluate_methods(dcg[0], dcg[1], [method])
			out.append((res[method], indim))
		return out
	else:
		out1, out2, out3 = [], [], []
		for dcg, indim in zip(data_chunks, indims):
			res = evaluate_methods(dcg[0], dcg[1], [method])
			out1.append((res['OR'], indim))
			out2.append((res['AND'], indim))
			out3.append((res['HC'], indim))
		return (out1, out2, out3)

def run_data_indim_L2(data_chunks, indims, method):
	out = []
	for dcg, indim in zip(data_chunks, indims):
		res = evaluate_methods_L2(dcg[0], dcg[1], [method])
		out.append((res[method], indim))
	return out

def run_data(data_chunks, noise_stds, method):
	if (method != 'FMPL'):
		out = []
		for dcg, noise_std in zip(data_chunks, noise_stds):
			res = evaluate_methods(dcg[0], dcg[1], [method])
			out.append((res[method], noise_std))
		return out
	else:
		out1, out2, out3 = [], [], []
		for dcg, noise_std in zip(data_chunks, noise_stds):
			res = evaluate_methods(dcg[0], dcg[1], [method])
			out1.append((res['OR'], noise_std))
			out2.append((res['AND'], noise_std))
			out3.append((res['HC'], noise_std))
		return (out1, out2, out3)

def run_data_L2(data_chunks, noise_stds, method):
	out = []
	for dcg, noise_std in zip(data_chunks, noise_stds):
		res = evaluate_methods_L2(dcg[0], dcg[1], [method])
		out.append((res[method], noise_std))
	return out

def run_noise_stds(dir_name, method):
	f = open(dir_name + '/' + 'params','r')
	data_params, init_args = pickle.load(f)
	f.close()
	indim, outdim, sample_size, repeats, noise_stds, seed = init_args
	raw_data = generate_seed_data(data_params, seed)
	data_chunks = []
	for i, noise_std in enumerate(noise_stds):
		data_chunk = []
		Graph = raw_data[i*repeats][1]
		for j in xrange(repeats):
			data_chunk.append(raw_data[i*repeats + j][0])
		data_chunks.append((data_chunk, Graph))
	if (method != 'FMPL'):
		out = run_data(data_chunks, noise_stds, method)
		save_data(out, method, dir_name)
	else:
		out1, out2, out3 = run_data(data_chunks, noise_stds, method)
		save_data(out1, 'OR', dir_name)
		save_data(out2, 'AND', dir_name)
		save_data(out3, 'HC', dir_name)

def run_noise_stds_L2(dir_name, method):
	f = open(dir_name + '/' + 'params','r')
	data_params, init_args = pickle.load(f)
	f.close()
	indim, outdim, sample_size, repeats, noise_stds, seed = init_args
	raw_data = generate_seed_data_L2(data_params, seed)
	data_chunks = []
	for i, noise_std in enumerate(noise_stds):
		data_chunk = []
		Graph = raw_data[i*repeats][2]
		for j in xrange(repeats):
			data_chunk.append((raw_data[i*repeats + j][0], raw_data[i*repeats + j][1]))
		data_chunks.append((data_chunk, Graph))
	return run_data_L2(data_chunks, noise_stds, method)

def run_outdims(dir_name, method):
	f = open(dir_name + '/' + 'params','r')
	data_params, init_args = pickle.load(f)
	f.close()
	indim, outdims, sample_size, repeats, noise_std, seed = init_args
	raw_data = generate_seed_data(data_params, seed)
	data_chunks = []
	for i, outdim in enumerate(outdims):
		data_chunk = []
		Graph = raw_data[i*repeats][1]
		for j in xrange(repeats):
			data_chunk.append(raw_data[i*repeats + j][0])
		data_chunks.append((data_chunk, Graph))
	if (method != 'FMPL'):
		out = run_data_outdim(data_chunks, outdims, method)
		save_data(out, method, dir_name)
	else:
		out1, out2, out3 = run_data_outdim(data_chunks, outdims, method)
		save_data(out1, 'OR', dir_name)
		save_data(out2, 'AND', dir_name)
		save_data(out3, 'HC', dir_name)

def run_outdims_L2(dir_name, method):
	f = open(dir_name + '/' + 'params','r')
	data_params, init_args = pickle.load(f)
	f.close()
	indim, outdims, sample_size, repeats, noise_std, seed = init_args
	raw_data = generate_seed_data_L2(data_params, seed)
	data_chunks = []
	for i, outdim in enumerate(outdims):
		data_chunk = []
		Graph = raw_data[i*repeats][2]
		for j in xrange(repeats):
			data_chunk.append((raw_data[i*repeats + j][0], raw_data[i*repeats + j][1]))
		data_chunks.append((data_chunk, Graph))
	return run_data_outdim_L2(data_chunks, outdims, method)

def init_outdims(indim = depa.default_indim, outdims = depa.default_outdims, sample_size = depa.default_sample_size, repeats=depa.default_repeat_count, noise_std = depa.default_noise_std, seed=depa.default_seed):
	data_params = []
	for outdim in outdims:
		for _ in xrange(repeats):
			data_params.append([indim, outdim, sample_size, noise_std])
	data_info = (data_params, [indim, outdims, sample_size, repeats, noise_std, seed])
	dir_name = '../test_data/test_outdims_n' + str(sample_size)
	os.mkdir(dir_name)
	f = open(dir_name + '/' + 'params', 'w')
	pickle.dump(data_info, f)
	f.close()
	return dir_name

def run_indims(dir_name, method):
	f = open(dir_name + '/' + 'params','r')
	data_params, init_args = pickle.load(f)
	f.close()
	indims, outdim, sample_size, repeats, noise_std, seed = init_args
	raw_data = generate_seed_data(data_params, seed)
	data_chunks = []
	for i, indim in enumerate(indims):
		data_chunk = []
		Graph = raw_data[i*repeats][1]
		for j in xrange(repeats):
			data_chunk.append(raw_data[i*repeats + j][0])
		data_chunks.append((data_chunk, Graph))
	if (method != 'FMPL'):
		out = run_data_indim(data_chunks, indims, method)
		save_data(out, method, dir_name)
	else:
		out1, out2, out3 = run_data_indim(data_chunks, indims, method)
		save_data(out1, 'OR', dir_name)
		save_data(out2, 'AND', dir_name)
		save_data(out3, 'HC', dir_name)

def run_indims_L2(dir_name, method):
	f = open(dir_name + '/' + 'params','r')
	data_params, init_args = pickle.load(f)
	f.close()
	indims, outdim, sample_size, repeats, noise_std, seed = init_args
	raw_data = generate_seed_data_L2(data_params, seed)
	data_chunks = []
	for i, indim in enumerate(indims):
		data_chunk = []
		Graph = raw_data[i*repeats][2]
		for j in xrange(repeats):
			data_chunk.append((raw_data[i*repeats + j][0], raw_data[i*repeats + j][1]))
		data_chunks.append((data_chunk, Graph))
	return run_data_indim_L2(data_chunks, indims, method)

def init_indims(indims = depa.default_indims, outdim = depa.default_outdim, sample_size = depa.default_sample_size, repeats=depa.default_repeat_count, noise_std = depa.default_noise_std, seed=depa.default_seed):
	data_params = []
	for indim in indims:
		for _ in xrange(repeats):
			data_params.append([indim, outdim, sample_size, noise_std])
	data_info = (data_params, [indims, outdim, sample_size, repeats, noise_std, seed])
	dir_name = '../test_data/test_indims_n' + str(sample_size)
	os.mkdir(dir_name)
	f = open(dir_name + '/' + 'params', 'w')
	pickle.dump(data_info, f)
	f.close()
	return dir_name

def init_noise_stds(indim, outdim, sample_size, repeats=depa.default_repeat_count, noise_stds = depa.default_noise_stds, seed=depa.default_seed):
	data_params = []
	for noise_std in noise_stds:
		for _ in xrange(repeats):
			data_params.append([indim, outdim, sample_size, noise_std])
	data_info = (data_params, [indim, outdim, sample_size, repeats, noise_stds, seed])
	dir_name = '../test_data/test_noise_stds_n' + str(sample_size)
	os.mkdir(dir_name)
	f = open(dir_name + '/' + 'params', 'w')
	pickle.dump(data_info, f)
	f.close()
	return dir_name

