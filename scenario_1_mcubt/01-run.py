"""
Batch simulation of mCUBT on scenario 1.

Generates N_SIM independent datasets using the same parameters as
scenario_1/01-data_generation.py and runs mCUBT on each.

Results saved to scenario_1_mcubt/results/results_mcubt.pkl as a list of dicts:
    [{'n_clusters': int, 'ARI': float, 'comp_time': float, 'root_k': int}, ...]

Run from the project root:
    python scenario_1_mcubt/01-run.py
"""

import multiprocessing
import os
import pickle
import sys
import time

import numpy as np
from joblib import Parallel, delayed
from sklearn.metrics import adjusted_rand_score

# --- Path setup ---
ROOT_DIR = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
sys.path.insert(0, ROOT_DIR)

from mcubt import MCUBT, MNode
from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.simulation.karhunen import KarhunenLoeve

# --- Directories ---
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Simulation parameters (identical to scenario_1) ---
N_SIM = 500          # number of independent replications
N_OBS = 300          # observations per KL draw
N_FEATURES = 3       # KL basis functions
N_CLUSTERS_TRUE = 2  # KL clusters (yields 5 true functional clusters)

CENTERS = np.array([[0, 0], [0, 0], [0, 0]])
CLUSTER_STD = np.array([[4, 1], [2.66, 0.66], [1.33, 0.33]])

# --- mCUBT hyper-parameters ---
N_COMPONENTS = 0.95
MIN_SIZE = 10
MAX_GROUP = 8
MIN_GROUP_SIZE = 10


def generate_data(seed: int):
    """Generate one scenario-1 dataset.

    Uses a fixed seed for reproducibility. The dataset has N_OBS * 5 / 2 = 750
    observations across 5 true functional clusters (labels 0-4).
    """
    np.random.seed(seed)
    argvals = {'input_dim_0': np.linspace(0, 1, 100)}

    simu = KarhunenLoeve(
        'wiener',
        n_functions=N_FEATURES,
        argvals=argvals,
    )
    simu.new(
        n_obs=N_OBS,
        n_clusters=N_CLUSTERS_TRUE,
        centers=CENTERS,
        cluster_std=CLUSTER_STD,
    )

    t = simu.data.argvals['input_dim_0']
    mean_1 = 20 / (1 + np.exp(-t))
    mean_2 = -25 / (1 + np.exp(-t))

    half = int(N_OBS / 2)
    v = simu.data.values

    new_values = np.vstack([
        v[:half]  + mean_1,   # cluster 0
        v[half:]  + mean_1,   # cluster 1
        v[:half]  + mean_2,   # cluster 2
        v[half:]  + mean_2,   # cluster 3
        v[half:]  + mean_2 - 15 * t,  # cluster 4
    ])

    data = DenseFunctionalData(simu.data.argvals, new_values)
    labels = np.hstack([
        simu.labels,
        simu.labels + 2,
        np.repeat(4, half),
    ])
    return data, labels


def run_one(idx: int) -> dict:
    """Run mCUBT on one generated dataset and return metrics."""
    print(f'  Simulation {idx:>4d}', flush=True)
    data_fd, labels = generate_data(seed=idx)

    t0 = time.time()
    root = MNode(data_fd, is_root=True)
    mcubt = MCUBT(root_node=root)
    mcubt.grow(
        n_components=N_COMPONENTS,
        min_size=MIN_SIZE,
        max_group=MAX_GROUP,
        min_group_size=MIN_GROUP_SIZE,
    )
    mcubt.join(n_components=N_COMPONENTS, max_group=MAX_GROUP)
    comp_time = time.time() - t0

    n_clusters = len(np.unique(mcubt.labels_join))
    ari = adjusted_rand_score(labels, mcubt.labels_join)

    # K̂ chosen at the root node (1 if root is immediately a leaf)
    root_k = mcubt.root_node.best_k if mcubt.root_node.best_k is not None else 1

    return {
        'n_clusters': n_clusters,
        'ARI': ari,
        'comp_time': comp_time,
        'root_k': root_k,
    }


def main():
    n_cores = multiprocessing.cpu_count()
    print(f'Running {N_SIM} mCUBT simulations on {n_cores} cores...')

    t0 = time.time()
    results = Parallel(n_jobs=n_cores)(
        delayed(run_one)(i) for i in range(N_SIM)
    )
    elapsed = time.time() - t0
    print(f'Done in {elapsed:.1f}s.')

    out_path = os.path.join(RESULTS_DIR, 'results_mcubt.pkl')
    with open(out_path, 'wb') as f:
        pickle.dump(results, f)
    print(f'Results saved to {out_path}')


if __name__ == '__main__':
    main()
