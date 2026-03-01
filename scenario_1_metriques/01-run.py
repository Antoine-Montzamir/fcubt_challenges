"""
Batch simulation of fCUBT with BIC / AIC / ICL / Stability on scenario 1.

Generates N_SIM independent datasets using the same parameters as
scenario_1/01-data_generation.py and runs all four variants on each.

Results saved to scenario_1_metriques/results/:
    results_bic.pkl
    results_aic.pkl
    results_icl.pkl
    results_stability.pkl

Each file is a list of dicts:
    [{'n_clusters': int, 'ARI': float, 'comp_time': float}, ...]

Run from the project root:
    python scenario_1_metriques/01-run.py
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

from fcubt2         import Node2,          FCUBT2
from fcubt_aic      import NodeAIC,        FCUBTAIC
from fcubt_icl      import NodeICL,        FCUBTICL
from fcubt_stability import NodeStability, FCUBTStability
from FDApy.representation.functional_data import DenseFunctionalData
from FDApy.simulation.karhunen import KarhunenLoeve

# --- Directories ---
RESULTS_DIR = os.path.join(os.path.dirname(os.path.abspath(__file__)), 'results')
os.makedirs(RESULTS_DIR, exist_ok=True)

# --- Simulation parameters (identical to scenario_1) ---
N_SIM = 200          # number of independent replications
N_OBS = 300          # observations per KL draw
N_FEATURES = 3       # KL basis functions
N_CLUSTERS_TRUE = 2  # KL clusters (yields 5 true functional clusters)

CENTERS = np.array([[0, 0], [0, 0], [0, 0]])
CLUSTER_STD = np.array([[4, 1], [2.66, 0.66], [1.33, 0.33]])

# --- fCUBT hyper-parameters ---
N_COMPONENTS = 0.95
MIN_SIZE = 10
MAX_GROUP = 5

# --- Stability-specific ---
N_BOOTSTRAPS   = 5
STAB_THRESHOLD = 0.5


def generate_data(seed: int):
    """Generate one scenario-1 dataset (5 true functional clusters)."""
    np.random.seed(seed)
    argvals = {'input_dim_0': np.linspace(0, 1, 100)}

    simu = KarhunenLoeve('wiener', n_functions=N_FEATURES, argvals=argvals)
    simu.new(n_obs=N_OBS, n_clusters=N_CLUSTERS_TRUE,
             centers=CENTERS, cluster_std=CLUSTER_STD)

    t = simu.data.argvals['input_dim_0']
    mean_1 = 20 / (1 + np.exp(-t))
    mean_2 = -25 / (1 + np.exp(-t))
    half = int(N_OBS / 2)
    v = simu.data.values

    new_values = np.vstack([
        v[:half]  + mean_1,           # cluster 0
        v[half:]  + mean_1,           # cluster 1
        v[:half]  + mean_2,           # cluster 2
        v[half:]  + mean_2,           # cluster 3
        v[half:]  + mean_2 - 15 * t,  # cluster 4
    ])

    data = DenseFunctionalData(simu.data.argvals, new_values)
    labels = np.hstack([simu.labels, simu.labels + 2, np.repeat(4, half)])
    return data, labels


# ---------------------------------------------------------------------------
# One runner per criterion
# ---------------------------------------------------------------------------

def run_one_bic(idx: int) -> dict:
    print(f'  [BIC]       Simulation {idx:>4d}', flush=True)
    data_fd, labels = generate_data(seed=idx)
    t0 = time.time()
    root = Node2(data_fd, is_root=True)
    model = FCUBT2(root_node=root)
    model.grow(n_components=N_COMPONENTS, min_size=MIN_SIZE, max_group=MAX_GROUP)
    model.join(n_components=N_COMPONENTS, max_group=MAX_GROUP)
    return {'n_clusters': len(np.unique(model.labels_join)),
            'ARI': adjusted_rand_score(labels, model.labels_join),
            'comp_time': time.time() - t0}


def run_one_aic(idx: int) -> dict:
    print(f'  [AIC]       Simulation {idx:>4d}', flush=True)
    data_fd, labels = generate_data(seed=idx)
    t0 = time.time()
    root = NodeAIC(data_fd, is_root=True)
    model = FCUBTAIC(root_node=root)
    model.grow(n_components=N_COMPONENTS, min_size=MIN_SIZE, max_group=MAX_GROUP)
    model.join(n_components=N_COMPONENTS, max_group=MAX_GROUP)
    return {'n_clusters': len(np.unique(model.labels_join)),
            'ARI': adjusted_rand_score(labels, model.labels_join),
            'comp_time': time.time() - t0}


def run_one_icl(idx: int) -> dict:
    print(f'  [ICL]       Simulation {idx:>4d}', flush=True)
    data_fd, labels = generate_data(seed=idx)
    t0 = time.time()
    root = NodeICL(data_fd, is_root=True)
    model = FCUBTICL(root_node=root)
    model.grow(n_components=N_COMPONENTS, min_size=MIN_SIZE, max_group=MAX_GROUP)
    model.join(n_components=N_COMPONENTS, max_group=MAX_GROUP)
    return {'n_clusters': len(np.unique(model.labels_join)),
            'ARI': adjusted_rand_score(labels, model.labels_join),
            'comp_time': time.time() - t0}


def run_one_stability(idx: int) -> dict:
    print(f'  [Stability] Simulation {idx:>4d}', flush=True)
    data_fd, labels = generate_data(seed=idx)
    t0 = time.time()
    root = NodeStability(data_fd, is_root=True,
                         n_bootstraps=N_BOOTSTRAPS,
                         stab_threshold=STAB_THRESHOLD,
                         stab_random_state=idx)
    model = FCUBTStability(root_node=root)
    model.grow(n_components=N_COMPONENTS, min_size=MIN_SIZE, max_group=MAX_GROUP)
    model.join(n_components=N_COMPONENTS, max_group=MAX_GROUP)
    return {'n_clusters': len(np.unique(model.labels_join)),
            'ARI': adjusted_rand_score(labels, model.labels_join),
            'comp_time': time.time() - t0}


def main():
    n_cores = multiprocessing.cpu_count()
    criteria = [
        ('bic',       run_one_bic),
        ('aic',       run_one_aic),
        ('icl',       run_one_icl),
        ('stability', run_one_stability),
    ]
    for name, runner in criteria:
        print(f'\n[{name.upper()}] Running {N_SIM} simulations on {n_cores} cores...')
        t0 = time.time()
        results = Parallel(n_jobs=n_cores)(
            delayed(runner)(i) for i in range(N_SIM)
        )
        print(f'[{name.upper()}] Done in {time.time() - t0:.1f}s.')
        out = os.path.join(RESULTS_DIR, f'results_{name}.pkl')
        with open(out, 'wb') as f:
            pickle.dump(results, f)
        print(f'[{name.upper()}] Saved to {out}')


if __name__ == '__main__':
    main()
