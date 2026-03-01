# # Perform FPCA+GMM on the data

# Load packages
import numpy as np
import pickle

from FDApy.preprocessing.dim_reduction.fpca import MFPCA
from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.representation.functional_data import MultivariateFunctionalData

from sklearn.mixture import GaussianMixture

# Load data
import os

os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
with open(os.path.join(DATA_DIR, 'canadian_smooth.pkl'), 'rb') as f:
    data_fd = pickle.load(f)

# Do MFPCA on the data for clustering
fpca = MFPCA(n_components=[0.99, 0.99])
fpca.fit(data_fd, method='NumInt')
    
# Compute scores
simu_proj = fpca.transform(data_fd)

results = {}
for i in range(2, 9):
    gm = GaussianMixture(i)
    gm.fit(simu_proj)
    bic = gm.bic(simu_proj)
    pred = gm.predict(simu_proj)
    results[i] = {'BIC': bic, 'labels': pred}

results[7]

with open('./results/results_weather_FPCA_GMM.pkl', 'wb') as f:
    pickle.dump(results, f)
