import numpy as np
import numpy.random as npr
import numpy.linalg as npl
import pandas as pd
import os
from sklearn.utils.extmath import fast_logdet

# Data available at: https://genome-cancer.ucsc.edu/proj/site/hgHeatmap/
# 1. Click "Add Datasets"
# 2. Select "TCGA breast invasive carcinoma"
# 3. Download "BRCA gene expression (IlluminaHiSeq)"
# 4. Extract the file "genomicMatrix" from the archive

def load_data():
    dt = pd.read_table('genomicMatrix', index_col=0)
    dt = dt.loc[dt.std(1) != 0,]
    return dt.values

def generate_example(m, dim, cutoff):
    done = False
    while not done:
        J = npr.choice(m.shape[0], dim, False)
        obscov = np.corrcoef(m[J,])
        obsprec = npl.inv(obscov)
        newprec = obsprec
        newprec[abs(newprec) < cutoff] = 0
        if fast_logdet(newprec) > -1e10:
            done = True
    newcov = npl.inv(newprec)
    return newprec, newcov
