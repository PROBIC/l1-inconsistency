# -*- coding: utf-8 -*-
import numpy as np
import numpy.linalg as npl
import matplotlib.pyplot as plt

from gene_expression_core import generate_example, load_data
from glasso_objective import glasso_objective_split
from sklearn.utils.extmath import fast_logdet

def glasso_objective_geneexp(m, d, cutoff=0.1, repeats=100):
    glassosplits = np.zeros((len(d), 6))
    truesplits = np.zeros((len(d), 6))
    for j, dim in enumerate(d):
        gsplits = np.zeros((repeats, 6))
        tsplits = np.zeros((repeats, 6))
        for i in range(repeats):
            newprec, newcov = generate_example(m, dim, cutoff)
            while fast_logdet(newprec) < -1e10:
                newprec, newcov = generate_example(m, dim, cutoff)
            ts, gs = glasso_objective_split(newprec, newcov)
            while gs[0] < -1e10:
                newprec, newcov = generate_example(m, dim, cutoff)
                while fast_logdet(newprec) < -1e10:
                    newprec, newcov = generate_example(m, dim, cutoff)
                ts, gs = glasso_objective_split(newprec, newcov)
            tsplits[i,:] = ts
            gsplits[i,:] = gs
        glassosplits[j,:] = np.mean(gsplits, 0)
        truesplits[j,:] = np.mean(tsplits, 0)
    return (glassosplits, truesplits)


def plot_geneexp_splits(gsplits, tsplits, t):
    plt.figure(figsize=(87.0/25.4, 70/25.4))
    plt.rcParams.update({'font.size': 7.0})
    plt.plot(t, (gsplits[:, 2])/t, "b-")
    plt.plot(t, (tsplits[:, 2])/t, "b--")
    plt.plot(t, (gsplits[:, 0] + gsplits[:, 1])/t, "g-")
    plt.plot(t, (tsplits[:, 0] + tsplits[:, 1])/t, "g--")
    plt.xlabel('$d$')
    plt.ylabel('$C/d$')
    plt.legend(["glasso penalty", "truth penalty", "glasso logl", "truth logl"], loc="center left")
    #gs = np.hstack((gsplits[:, 0:1] + gsplits[:, 1:2], gsplits[:, 2:3]))
    #ts = np.hstack((tsplits[:, 0:1] + tsplits[:, 1:2], tsplits[:, 2:3]))
    #f, (ax1, ax2) = plt.subplots(1, 2, sharey=True)
    #ax1.plot(t, gs / t[:, np.newaxis])
    #ax2.plot(t, ts / t[:, np.newaxis])


def do_plot_geneexp_splits(m):
    d = np.arange(10, 110, 10)
    gsplits, tsplits = glasso_objective_geneexp(m, d, repeats=10)
    plot_geneexp_splits(gsplits, tsplits, d)
