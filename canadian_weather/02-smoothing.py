# # Smoothing of the data

# Load packages
import numpy as np
import pickle

from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.representation.functional_data import MultivariateFunctionalData

# Load data
import os

os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
with open(os.path.join(DATA_DIR, 'canadian_temperature_daily_reduced.pkl'), 'rb') as f:
    temperature = pickle.load(f)
with open(os.path.join(DATA_DIR, 'canadian_precipitation_daily_reduced.pkl'), 'rb') as f:
    precipitation = pickle.load(f)

# Smoothing of the data
temperature_smooth = temperature.smooth(points=0.5, neighborhood=4)
precipitation_smooth = precipitation.smooth(points=0.5, neighborhood=2)

# Create multivariate functional data
data_fd = MultivariateFunctionalData([temperature_smooth, precipitation_smooth])

# Save the reduced data
with open(os.path.join(DATA_DIR, 'canadian_smooth.pkl'), 'wb') as f:
    pickle.dump(data_fd, f)
