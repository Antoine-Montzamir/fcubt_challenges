# # Preprocessing of the data

# Load packages
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pickle

from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.misc.loader import read_csv
from FDApy.visualization.plot import plot

import os

os.makedirs(os.path.join(os.path.dirname(__file__), 'figures'), exist_ok=True)
FIGURES_DIR = os.path.join(os.path.dirname(__file__), 'figures')
os.makedirs(os.path.join(os.path.dirname(__file__), 'data'), exist_ok=True)
DATA_DIR = os.path.join(os.path.dirname(__file__), 'data')
matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

# Load data
temperature = read_csv(os.path.join(DATA_DIR, 'canadian_temperature_daily.csv'), index_col=0)
temperature = DenseFunctionalData({'input_dim_0': np.linspace(0, 1, num=365)}, temperature.values)

precipitation = read_csv(os.path.join(DATA_DIR, 'canadian_precipitation_daily.csv'), index_col=0)
precipitation = DenseFunctionalData({'input_dim_0': np.linspace(0, 1, num=364)}, precipitation.values)

_ = plot(temperature)
_ = plt.xlabel('$t$')
_ = plt.ylabel('Temperature')
plt.savefig(os.path.join(FIGURES_DIR, 'temperature.pdf'))

_ = plot(precipitation)
_ = plt.xlabel('$t$')
_ = plt.ylabel('Precipitation')
plt.savefig(os.path.join(FIGURES_DIR, 'precipitation.pdf'))

# We normalize the data following the methodology using by Jacques and Preda, in Model-based clustering for multivariate functional data (2012), section 2.3.

# Data reduction
cov_temp = temperature.covariance()
reduced_temp = DenseFunctionalData(temperature.argvals, temperature.values / np.sqrt(np.diag(cov_temp.values[0])))

cov_precipitation = precipitation.covariance()
reduced_prec = DenseFunctionalData(precipitation.argvals, precipitation.values / np.sqrt(np.diag(cov_precipitation.values[0])))

# Save the reduced data
with open(os.path.join(DATA_DIR, 'canadian_temperature_daily_reduced.pkl'), 'wb') as f:
    pickle.dump(reduced_temp, f)
with open(os.path.join(DATA_DIR, 'canadian_precipitation_daily_reduced.pkl'), 'wb') as f:
    pickle.dump(reduced_prec, f)

# Save as CSV for R methods
np.savetxt(os.path.join(DATA_DIR, 'canadian_temperature_daily_reduced.csv'), reduced_temp.values, delimiter=',')
np.savetxt(os.path.join(DATA_DIR, 'canadian_precipitation_daily_reduced.csv'), reduced_prec.values, delimiter=',')
