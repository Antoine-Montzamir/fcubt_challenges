#!/usr/bin/env python
# -*-coding:utf8 -*

"""
fCUBT with bootstrap Stability model selection.

Identical to fcubt2 (binary tree, GMM(2) split) except that the number of
components K is chosen by maximising bootstrap stability instead of BIC.

For each K >= 2:
    1. Fit GMM(K) on the full scores → reference labels.
    2. For B bootstrap resamples: re-fit GMM(K), predict on original data,
       compute ARI vs reference labels.
    3. Stab(K) = mean ARI.
Selects K = argmax Stab(K) if the best score >= stab_threshold, else K=1.

Usage
-----
    from fcubt_stability import NodeStability, FCUBTStability

    root  = NodeStability(data_fd, is_root=True,
                          n_bootstraps=10, stab_threshold=0.5)
    model = FCUBTStability(root_node=root)
    model.grow(n_components=0.95, min_size=10, max_group=5)
    model.join(n_components=0.95, max_group=5)
    labels = model.labels_join
"""

import numpy as np
from sklearn.metrics import adjusted_rand_score
from sklearn.mixture import GaussianMixture

from fcubt2 import Node2, FCUBT2


class NodeStability(Node2):
    """Node2 variant that selects K by bootstrap stability.

    Parameters
    ----------
    n_bootstraps : int, default=10
        Number of bootstrap resamples per K.
    stab_threshold : float, default=0.5
        Minimum stability to accept a split (otherwise K=1 → leaf).
    stab_random_state : int, default=0
        Seed for the bootstrap RNG.
    """

    def __init__(
        self,
        *args,
        n_bootstraps: int = 10,
        stab_threshold: float = 0.5,
        stab_random_state: int = 0,
        **kwargs,
    ) -> None:
        super().__init__(*args, **kwargs)
        self.n_bootstraps = n_bootstraps
        self.stab_threshold = stab_threshold
        self.stab_random_state = stab_random_state

    def _select_k(self, scores: np.ndarray, k_range: np.ndarray) -> int:
        rng = np.random.RandomState(self.stab_random_state)
        n = len(scores)
        ks = [int(k) for k in k_range if int(k) >= 2]
        if not ks:
            return 1

        best_k, best_stab = 1, -np.inf
        for k in ks:
            labels0 = GaussianMixture(n_components=k, random_state=0).fit_predict(scores)
            aris = []
            for b in range(self.n_bootstraps):
                idx = rng.choice(n, size=n, replace=True)
                try:
                    gm_b = GaussianMixture(n_components=k, random_state=int(b))
                    gm_b.fit(scores[idx])
                    aris.append(adjusted_rand_score(labels0, gm_b.predict(scores)))
                except Exception:
                    aris.append(0.0)
            stab = float(np.mean(aris))
            if stab > best_stab:
                best_stab, best_k = stab, k

        return best_k if best_stab >= self.stab_threshold else 1

    def _make_child(self, data, identifier, idx_obs) -> 'NodeStability':
        """Propagate stability parameters to child nodes."""
        return type(self)(
            data, identifier=identifier, idx_obs=idx_obs,
            normalize=self.normalize,
            n_bootstraps=self.n_bootstraps,
            stab_threshold=self.stab_threshold,
            stab_random_state=self.stab_random_state,
        )


class FCUBTStability(FCUBT2):
    """fCUBT2 using bootstrap Stability for model selection at each node."""
