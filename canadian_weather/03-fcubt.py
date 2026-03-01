# # Perform fCUBT on the data

# Load packages
import numpy as np
import pickle

from FDApy.clustering.fcubt import Node, FCUBT
from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.representation.functional_data import MultivariateFunctionalData

# Load data
import os

os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
with open(os.path.join(DATA_DIR, 'canadian_smooth.pkl'), 'rb') as f:
    data_fd = pickle.load(f)

# Build the tree
root_node = Node(data_fd, is_root=True)
fcubt = FCUBT(root_node=root_node)

# Growing
fcubt.grow(n_components=[0.95, 0.95], min_size=10)

fcubt.labels

with open('./results/results_weather_growing.pkl', 'wb') as f:
    pickle.dump(fcubt.labels, f)

# Joining
final_labels = fcubt.join(n_components=[0.95, 0.95])

final_labels

with open('./results/results_weather_fcubt.pkl', 'wb') as f:
    pickle.dump(final_labels, f)
