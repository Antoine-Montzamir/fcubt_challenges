#!/usr/bin/env python
# -*-coding:utf8 -*

"""
fCUBT with ICL model selection.

Identical to fcubt2 (binary tree, GMM(2) split) except that the number of
components K is chosen by minimising ICL instead of BIC.

ICL(K) = BIC(K) - 2 * sum_ik gamma_ik * log(gamma_ik)

The entropy term penalises uncertain assignments, so ICL tends to select
fewer, better-separated clusters than BIC.

Usage
-----
    from fcubt_icl import NodeICL, FCUBTICL

    root  = NodeICL(data_fd, is_root=True)
    model = FCUBTICL(root_node=root)
    model.grow(n_components=0.95, min_size=10, max_group=5)
    model.join(n_components=0.95, max_group=5)
    labels = model.labels_join
"""

import numpy as np
from sklearn.mixture import GaussianMixture

from fcubt2 import Node2, FCUBT2


class NodeICL(Node2):
    """Node2 variant that selects K by minimising ICL."""

    def _select_k(self, scores: np.ndarray, k_range: np.ndarray) -> int:
        best_k, best_icl = 1, np.inf
        for k in k_range:
            gm = GaussianMixture(n_components=int(k), random_state=0)
            gm.fit(scores)
            bic = gm.bic(scores)
            gamma = gm.predict_proba(scores)
            log_gamma = np.log(np.where(gamma > 0, gamma, 1.0))
            icl = bic - 2.0 * np.sum(gamma * log_gamma)
            if icl < best_icl:
                best_icl, best_k = icl, int(k)
        return best_k


class FCUBTICL(FCUBT2):
    """fCUBT2 using ICL for model selection at each node."""
