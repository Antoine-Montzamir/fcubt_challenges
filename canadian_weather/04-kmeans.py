# # Perform $k$-means on the data

# Load packages
import numpy as np
import pickle

from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.representation.functional_data import MultivariateFunctionalData

from skfda import FDataGrid
from skfda.ml.clustering import KMeans

# Load data
import os

os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
with open(os.path.join(DATA_DIR, 'canadian_smooth.pkl'), 'rb') as f:
    data_fd = pickle.load(f)

# Format data for skfda
temperature = data_fd[0].values
precipitation = data_fd[1].values

# skfda only accept data with same shape
new_prec = np.hstack([precipitation,
                      precipitation[:, -1][:, np.newaxis]])

# Create FDataGrid object
data_matrix = np.stack([temperature, new_prec], axis=-1)
sample_points = data_fd[0].argvals['input_dim_0']
fdata = FDataGrid(data_matrix, sample_points)

# Compute derivatives
fdata_derivatives = fdata.derivative(order=1)

# Perform k-means
res = {}
for i in np.arange(2, 9, 1):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(fdata)
    res[i] = kmeans.predict(fdata)

res[4]

with open('./results/results_weather_kmeans_d1.pkl', 'wb') as f:
    pickle.dump(res, f)

# Perform k-means on derivatives
res_derivative = {}
for i in np.arange(2, 9, 1):
    kmeans = KMeans(n_clusters=i)
    kmeans.fit(fdata_derivatives)
    res_derivative[i] = kmeans.predict(fdata_derivatives)

res_derivative[4]

with open('./results/results_weather_kmeans_d2.pkl', 'wb') as f:
    pickle.dump(res_derivative, f)
