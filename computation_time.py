# Load packages
import matplotlib
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
import pickle
import pyreadr
import seaborn as sns

from sklearn.metrics import adjusted_rand_score, adjusted_mutual_info_score, v_measure_score
from sklearn.mixture import GaussianMixture
from sklearn.preprocessing import StandardScaler

import rpy2.robjects as robjects
from rpy2.robjects import pandas2ri

from matplotlib import colors as mcolors
COLORS = [v for v in mcolors.BASE_COLORS.values()]

readRDS = robjects.r['readRDS']

PATH_RESULTS = './results/'

matplotlib.use("pgf")
matplotlib.rcParams.update({
    "pgf.texsystem": "pdflatex",
    'font.family': 'serif',
    'text.usetex': True,
    'pgf.rcfonts': False,
})

COLORS = ["#377eb8", "#ff7f00", "#4daf4a",
          "#f781bf", "#a65628", "#984ea3",
          "#999999", "#e41a1c", "#dede00"]
custom_palette = sns.set_palette(sns.color_palette(COLORS))

# Load results
with open('./scenario_1/results/results_fcubt_comptime.pkl', 'rb') as f:
    comptime_1 = pickle.load(f)
results_comptime_1 = np.array([simu['comp_time'] for idx, simu in enumerate(comptime_1)])
with open('./scenario_2/results/results_fcubt_comptime.pkl', 'rb') as f:
    comptime_2 = pickle.load(f)
results_comptime_2 = np.array([simu['comp_time'] for idx, simu in enumerate(comptime_2)])
with open('./scenario_3/results/results_fcubt_comptime.pkl', 'rb') as f:
    comptime_3 = pickle.load(f)
results_comptime_3 = np.array([simu['comp_time'] for idx, simu in enumerate(comptime_3)])
with open('./scenario_4/results/results_fcubt_comptime.pkl', 'rb') as f:
    comptime_4 = pickle.load(f)
results_comptime_4 = np.array([simu['comp_time'] for idx, simu in enumerate(comptime_4)])

comptime = pd.DataFrame({'Scenario 1': results_comptime_1,
                         'Scenario 2': results_comptime_2,
                         'Scenario 3': results_comptime_3,
                         'Scenario 4': results_comptime_4,
                        })

plt.figure(figsize=(7, 5), constrained_layout=True)
bplot = sns.boxplot(data=comptime, orient='h')
bplot.set_yticklabels(bplot.get_yticklabels(), size=15)
for i in range(4):
    mybox = bplot.artists[i]
    mybox.set_facecolor(COLORS[i])
plt.xlabel('Computation time (in seconds)', size=16)
plt.savefig('./comptime.eps', format='eps')
