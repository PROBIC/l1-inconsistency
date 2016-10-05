# -*- coding: utf-8 -*-
import numpy as np
import matplotlib.pyplot as plt

from gene_expression_core import generate_example, load_data
import assumpt1 as as1

def test_assumption1(m, dim, CUTOFF=0.1):
    newprec, newcov = generate_example(m, dim, CUTOFF)
    sparsity = np.mean(newprec == 0)
    graph = 1.0 * (newprec != 0)
    #x = as1.assumption1_norm(obscov, graph)
    y = as1.assumption1_norm(newcov, graph)
    return y, sparsity

def test_several(m, dims, repeats=100, CUTOFF=0.1):
    values = np.zeros((repeats, len(dims)))
    sparsities = np.zeros((repeats, len(dims)))
    for i, d in enumerate(dims):
        for j in range(repeats):
            (values[j,i], sparsities[j,i]) = test_assumption1(m, d, CUTOFF)
    return values, sparsities

def plot_tests(d, res, assumpt_cutoffs=1.0, CUTOFF=0.1):
    plt.figure(figsize=(87.0/25.4, 70/25.4))
    plt.rcParams.update({'font.size': 7.0})
    for i, c in enumerate(assumpt_cutoffs):
        plt.plot(d, np.mean(res<c, 0), label='$c=%.1d$' % c)
    plt.xlabel('$d$')
    plt.ylabel('Fraction with $\gamma < c$')
    plt.legend(loc='upper right')
    plt.axis((5, 100, 0, 1))
    plt.xticks((5, 20, 40, 60, 80, 100))

# m = load_data()
# d = np.arange(5, 55, 5)
# d = np.append(5, np.arange(10, 110, 10))
# res, sparsities = test_several(m, d, repeats=30)
# plot_tests(d, res, [1.0, 3.0, 5.0, 7.0, 10.0, 15.0])
# plt.savefig('../figures/tcga_assumption1.pdf', bbox_inches='tight')
