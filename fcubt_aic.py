#!/usr/bin/env python
# -*-coding:utf8 -*

"""
fCUBT with AIC model selection.

Identical to fcubt2 (binary tree, GMM(2) split) except that the number of
components K is chosen by minimising AIC instead of BIC.

Usage
-----
    from fcubt_aic import NodeAIC, FCUBTAIC

    root  = NodeAIC(data_fd, is_root=True)
    model = FCUBTAIC(root_node=root)
    model.grow(n_components=0.95, min_size=10, max_group=5)
    model.join(n_components=0.95, max_group=5)
    labels = model.labels_join
"""

import numpy as np
from sklearn.mixture import GaussianMixture

from fcubt2 import Node2, FCUBT2


class NodeAIC(Node2):
    """Node2 variant that selects K by minimising AIC."""

    def _select_k(self, scores: np.ndarray, k_range: np.ndarray) -> int:
        best_k, best_aic = 1, np.inf
        for k in k_range:
            gm = GaussianMixture(n_components=int(k), random_state=0)
            gm.fit(scores)
            aic = gm.aic(scores)
            if aic < best_aic:
                best_aic, best_k = aic, int(k)
        return best_k


class FCUBTAIC(FCUBT2):
    """fCUBT2 using AIC for model selection at each node."""
