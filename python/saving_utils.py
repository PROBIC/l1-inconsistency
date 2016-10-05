import matplotlib.pyplot as plt
from matplotlib.backends.backend_pdf import PdfPages
from matplotlib import rc
rc('text', usetex=True)
import pickle

label_map = {"naive" : "naive", \
			"glasso" : "glasso", \
			"scio" : "scio", \
			"clime" : "clime", \
			"OR" : "FMPL", \
			"random": "random"}

color_map = {"naive" : "b", \
			"glasso" : "r", \
			"scio" : "m", \
			"clime" : "k", \
			"OR" : "c", \
			"random" : "g"}

def get_data(method, dir_name):
	f = open(dir_name + '/' + method, 'r')
	data = pickle.load(f)
	f.close()
	return data

def get_data_param(method, dir_name):
	return get_data(method, dir_name)[1]

def save_data(data, method, dir_name):
	f = open(dir_name + '/' + method, 'w')
	pickle.dump(data, f)
	f.close()

def save_figure(fig, name):
	figure_name = "../figures/" + name + ".pdf"
	pd = PdfPages(figure_name)
	pd.savefig(fig)
	pd.close()

def make_figuree(data_set, xs, ys, x_variable_name = None, y_variable_name = None, figure_title = "test_figure", dim = 144):
	if (x_variable_name == None):
		x_variable_name = xs
	if (y_variable_name == None):
		y_variable_name = ys
	fig = plt.figure()
	plt.title(figure_title)
	fontsize = 20
	plt.xlabel(x_variable_name, fontsize = fontsize)
	plt.ylabel(y_variable_name, fontsize = fontsize)
	outdims = []
	indims = []
	if (y_variable_name == "hamming distance"):
		ax = plt.subplot(111)
		if (xs == "outdim"):
			plt.ylabel("hamming distance $/ d_2$")
			for method in data_set:
				outdims = data_set[method][xs]
		if (xs == "indim"):
			plt.ylabel("hamming distance")
			for method in data_set:
				indims = data_set[method][xs]
	for method in data_set:
		yy = data_set[method][ys]
		if (xs == "outdim"):
			yy = [1.0*yyy/dim for yyy, dim in zip(yy, outdims)]
		m = len(data_set[method][xs])
		x = range(m)
		curve_label = label_map[method]
		curve_color = color_map[method]
		plt.plot(x, yy, label = curve_label, color=curve_color)
		plt.xticks(x, data_set[method][xs])
	plt.legend()
	if ((y_variable_name == "hamming distance") and (xs == "noise_stds")):
		plt.gca().set_ylim([0, 35])
	return fig
