#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Functional CUBT — standalone re-implementation (fcubt2).

Implements the fCUBT algorithm from scratch, following the same structure
as mcubt.py but preserving the original fCUBT behaviour:
  - BIC selects K̂ at each node.
  - If K̂ > 1: binary split using GMM(2) → left / right children.
  - Identifier scheme: (depth, position), matching the original fCUBT.
  - Joining step: greedy merging of non-sibling leaf pairs whose union
    is better described by K=1 than K=2 (BIC criterion).

Key differences from mCUBT
---------------------------
1. Split is always binary (GMM(2)), regardless of K̂ chosen by BIC.
2. Node identity: (depth, position) pair — siblings are (d, 2k)/(d, 2k+1).
3. Children: Node2.left / Node2.right instead of Node2.children list.

Usage
-----
    from fcubt2 import Node2, FCUBT2

    root  = Node2(data_fd, is_root=True)
    model = FCUBT2(root_node=root)
    model.grow(n_components=0.95, min_size=10, max_group=5)
    model.join(n_components=0.95, max_group=5)
    labels = model.labels_join
"""

import itertools
from typing import Dict, List, Optional, Set, Tuple, TypeVar, Union

import networkx as nx
import numpy as np
from sklearn.mixture import GaussianMixture

from FDApy.clustering.fcubt import format_label
from FDApy.clustering.optimalK.bic import BIC
from FDApy.preprocessing.dim_reduction.fcp_tpa import FCPTPA
from FDApy.preprocessing.dim_reduction.fpca import MFPCA, UFPCA
from FDApy.representation.functional_data import (DenseFunctionalData,
                                                   MultivariateFunctionalData)

N = TypeVar('N', bound='Node2')
T = TypeVar('T', bound='DenseFunctionalData')
M = TypeVar('M', bound='MultivariateFunctionalData')


###############################################################################
# Node2

class Node2:
    """A node in the fCUBT2 binary tree.

    Parameters
    ----------
    data : DenseFunctionalData or MultivariateFunctionalData
        Functional data held by this node.
    identifier : tuple (depth, position), default=(0, 0)
        - Root node          : (0, 0)
        - Left child of (d, p) : (d+1, 2*p)
        - Right child of (d, p): (d+1, 2*p+1)
        - Merged node (join)   : list of constituent identifiers.
    idx_obs : np.ndarray, default=None
        Global observation indices tracked through splits.
    is_root : bool, default=False
    normalize : bool, default=False
        Normalize data before FPCA.

    Attributes
    ----------
    left, right : Node2 or None
        Children created during split.
    labels : np.ndarray
        GMM(2) assignment (0 or 1) for each observation in this node.
    fpca : FPCA model
        Fitted dimensionality-reduction model.
    gaussian_model : GaussianMixture
        Fitted GMM(2) model used for the binary split.
    """

    def __init__(
        self,
        data: Union[T, M],
        identifier: Union[tuple, list] = (0, 0),
        idx_obs: Optional[np.ndarray] = None,
        is_root: bool = False,
        is_leaf: bool = False,
        normalize: bool = False,
    ) -> None:
        if not isinstance(data, (DenseFunctionalData, MultivariateFunctionalData)):
            raise TypeError("data must be DenseFunctionalData or MultivariateFunctionalData.")
        self.data = data
        self.identifier = (0, 0) if is_root else identifier
        self.idx_obs = np.arange(data.n_obs) if idx_obs is None else idx_obs
        self.labels = np.zeros(data.n_obs, dtype=int)
        self.is_root = is_root
        self.is_leaf = is_leaf
        self.normalize = normalize
        self.fpca = None
        self.gaussian_model = None
        self.left: Optional[N] = None
        self.right: Optional[N] = None

    # ------------------------------------------------------------------
    # Properties

    @property
    def depth(self) -> int:
        """Depth of this node (root = 0)."""
        ident = self.identifier
        if isinstance(ident, tuple):
            return ident[0]
        if isinstance(ident, list) and ident:
            first = ident[0]
            return first[0] if isinstance(first, tuple) else 0
        return 0

    # ------------------------------------------------------------------
    # Repr

    def __str__(self) -> str:
        return (f"Node2(id={self.identifier}, depth={self.depth}, "
                f"n_obs={self.data.n_obs}, is_root={self.is_root}, "
                f"is_leaf={self.is_leaf})")

    def __repr__(self) -> str:
        return self.__str__()

    # ------------------------------------------------------------------
    # Internal helpers

    def _compute_scores(
        self,
        n_components: Union[float, int],
    ) -> Tuple[np.ndarray, object]:
        """Fit FPCA on node data and return (scores, fpca_model)."""
        data = self.data
        if isinstance(data, DenseFunctionalData):
            if data.n_dim == 1:
                fpca = UFPCA(n_components=n_components, normalize=self.normalize)
                fpca.fit(data=data, method='GAM')
                scores = fpca.transform(data=data, method='NumInt')
            elif data.n_dim == 2:
                n_pts = data.n_points
                mv = np.diff(np.identity(n_pts['input_dim_0']))
                mw = np.diff(np.identity(n_pts['input_dim_1']))
                fpca = FCPTPA(n_components=n_components)
                fpca.fit(
                    data,
                    penal_mat={'v': mv @ mv.T, 'w': mw @ mw.T},
                    alpha_range={'v': np.array([1e-4, 1e4]),
                                 'w': np.array([1e-4, 1e4])},
                    tol=1e-4, max_iter=15, adapt_tol=True,
                )
                scores = fpca.transform(data)
            else:
                raise ValueError("Data dimension must be 1 or 2.")
        elif isinstance(data, MultivariateFunctionalData):
            fpca = MFPCA(n_components=n_components, normalize=self.normalize)
            fpca.fit(data=data, method='NumInt')
            scores = fpca.transform(data, method='NumInt')
        else:
            raise TypeError("Wrong data type.")
        return scores, fpca

    def _subset_data(self, mask: np.ndarray) -> Union[T, M]:
        """Subset node data with a boolean mask."""
        if isinstance(self.data, DenseFunctionalData):
            return self.data[mask]
        if isinstance(self.data, MultivariateFunctionalData):
            return MultivariateFunctionalData([obj[mask] for obj in self.data])
        raise TypeError("Wrong data type.")

    # ------------------------------------------------------------------
    # Core methods

    def split(
        self,
        n_components: Union[float, int] = 0.95,
        min_size: int = 10,
        max_group: int = 5,
    ) -> None:
        """Split this node into two children using BIC + GMM(2).

        BIC selects K̂ over {1, ..., max_group}. If K̂ > 1, a GMM(2) is
        fitted and observations are assigned to left / right. If K̂ = 1,
        this node becomes a leaf.

        Parameters
        ----------
        n_components : float or int, default=0.95
            FPCA components (or variance fraction) to retain.
        min_size : int, default=10
            Minimum observations required to attempt a split.
        max_group : int, default=5
            Maximum K to evaluate in BIC search.
        """
        if self.data.n_obs <= min_size:
            self.is_leaf = True
            return

        scores, fpca = self._compute_scores(n_components)
        self.fpca = fpca

        # --- Model selection (overridable via _select_k) ---
        k_range = np.arange(1, min(max_group, self.data.n_obs))
        best_k = self._select_k(scores, k_range)

        if best_k <= 1:
            self.is_leaf = True
            return

        # --- Binary split with GMM(2) ---
        gm = GaussianMixture(n_components=2, random_state=0)
        prediction = gm.fit_predict(scores)

        self.gaussian_model = gm
        self.labels = prediction

        d, p = self.identifier
        self.left  = self._make_child(self._subset_data(prediction == 0),
                                      (d + 1, 2 * p),
                                      self.idx_obs[prediction == 0])
        self.right = self._make_child(self._subset_data(prediction == 1),
                                      (d + 1, 2 * p + 1),
                                      self.idx_obs[prediction == 1])

    def _select_k(self, scores: np.ndarray, k_range: np.ndarray) -> int:
        """Select optimal K from FPCA scores. Override in subclasses."""
        bic_stat = BIC(parallel_backend=None)
        return bic_stat(scores, k_range)

    def _make_child(self, data, identifier, idx_obs) -> N:
        """Instantiate a child node. Override in subclasses to pass extra params."""
        return type(self)(data=data, identifier=identifier,
                         idx_obs=idx_obs, normalize=self.normalize)

    def unite(self, node: N) -> N:
        """Merge this node with `node` into a single Node2.

        The resulting node's identifier is a list of the two constituent
        identifiers, allowing chained merges.

        Parameters
        ----------
        node : Node2

        Returns
        -------
        Node2
        """
        data = self.data.concatenate(node.data)

        self_ids = self.identifier if isinstance(self.identifier, list) else [self.identifier]
        node_ids = node.identifier if isinstance(node.identifier, list) else [node.identifier]
        new_id = self_ids + node_ids

        return Node2(
            data=data,
            identifier=new_id,
            idx_obs=np.hstack([self.idx_obs, node.idx_obs]),
            is_root=(self.is_root and node.is_root),
            is_leaf=(self.is_leaf and node.is_leaf),
            normalize=self.normalize,
        )

    def predict(self, new_obs: Union[T, M]) -> N:
        """Route a new observation to its left or right child.

        Parameters
        ----------
        new_obs : FunctionalData
            A single new observation.

        Returns
        -------
        Node2 — the left or right child.
        """
        score = self.fpca.transform(new_obs, method='NumInt')
        pred = int(self.gaussian_model.predict(score)[0])
        return self.left if pred == 0 else self.right

    def predict_proba(self, new_obs: Union[T, M]) -> np.ndarray:
        """Return GMM(2) probabilities for a new observation.

        Returns
        -------
        np.ndarray of shape (1, 2)
        """
        score = self.fpca.transform(new_obs, method='NumInt')
        return self.gaussian_model.predict_proba(score)


###############################################################################
# Joining step

def joining_step2(
    list_nodes: List[N],
    siblings: Set[Tuple[N, N]],
    n_components: Union[int, float] = 0.95,
    max_group: int = 5,
    normalize: bool = False,
) -> List[N]:
    """One round of the fCUBT2 joining step.

    Tests all non-sibling leaf pairs. Merges the pair with the smallest BIC
    among those whose union is best described by a single Gaussian (K̂ = 1).

    Parameters
    ----------
    list_nodes : list of Node2
        Current set of leaf nodes.
    siblings : set of (Node2, Node2)
        Sibling pairs to exclude from merging candidates.
    n_components : float or int
        FPCA components for the joint dimensionality reduction.
    max_group : int
        Maximum K in BIC search.
    normalize : bool
        Normalize data before FPCA.

    Returns
    -------
    list of Node2 — updated leaf nodes after (at most) one merge.
    """
    nodes_combinations = set(itertools.combinations(list_nodes, 2))
    edges = nodes_combinations - siblings

    graph = nx.Graph()
    graph.add_nodes_from(list_nodes)
    graph.add_edges_from(edges)

    edges_to_remove = []
    for node1, node2 in graph.edges:
        new_data = node1.data.concatenate(node2.data)

        # FPCA on combined data
        if isinstance(new_data, DenseFunctionalData):
            if new_data.n_dim == 1:
                ufpca = UFPCA(n_components=n_components, normalize=normalize)
                ufpca.fit(data=new_data, method='GAM')
                scores = ufpca.transform(data=new_data, method='NumInt')
            elif new_data.n_dim == 2:
                n_pts = new_data.n_points
                pv = np.diff(np.identity(n_pts['input_dim_0']))
                pw = np.diff(np.identity(n_pts['input_dim_1']))
                fcptpa = FCPTPA(n_components=n_components)
                fcptpa.fit(
                    new_data,
                    penal_mat={'v': pv @ pv.T, 'w': pw @ pw.T},
                    alpha_range={'v': np.array([1e-4, 1e4]),
                                 'w': np.array([1e-4, 1e4])},
                    tol=1e-4, max_iter=15, adapt_tol=True,
                )
                scores = fcptpa.transform(new_data)
            else:
                raise ValueError("Data dimension must be 1 or 2.")
        elif isinstance(new_data, MultivariateFunctionalData):
            mfpca = MFPCA(n_components=n_components, normalize=normalize)
            mfpca.fit(data=new_data, method='NumInt')
            scores = mfpca.transform(new_data)
        else:
            raise TypeError("Wrong data type.")

        eff_max = min(max_group, new_data.n_obs)
        bic_stat = BIC(parallel_backend=None)
        best_k = bic_stat(scores, np.arange(1, eff_max))

        if best_k > 1:
            edges_to_remove.append((node1, node2))
        else:
            graph[node1][node2]['bic'] = bic_stat.bic_df['bic_value'].min()

    graph.remove_edges_from(edges_to_remove)

    if graph.number_of_edges() != 0:
        bic_dict = nx.get_edge_attributes(graph, 'bic')
        pair = min(bic_dict, key=bic_dict.get)
        merged = pair[0].unite(pair[1])
        graph.add_node(merged)
        graph.remove_node(pair[0])
        graph.remove_node(pair[1])

    return list(graph.nodes)


###############################################################################
# FCUBT2

class FCUBT2:
    """Functional CUBT — standalone binary-tree implementation.

    Parameters
    ----------
    root_node : Node2
        Root node of the tree (wraps the full dataset).
    normalize : bool, default=False
        Normalize data before FPCA at each node.

    Attributes
    ----------
    tree : list of Node2
        All nodes (internal + leaves) in BFS order.
    labels_grow : np.ndarray
        Cluster labels after the growing step.
    labels_join : np.ndarray
        Cluster labels after the joining step.
    mapping_grow, mapping_join : dict
        Mapping from leaf Node2 → integer cluster label.
    """

    def __init__(
        self,
        root_node: Optional[N] = None,
        normalize: bool = False,
    ) -> None:
        self.root_node = root_node
        self.tree: List[N] = [root_node]
        self.normalize = normalize

    # ------------------------------------------------------------------
    # Properties

    @property
    def n_nodes(self) -> int:
        return len(self.tree)

    @property
    def n_leaf(self) -> int:
        return sum(1 for n in self.tree if n.is_leaf)

    @property
    def height(self) -> int:
        return max(n.depth for n in self.tree) + 1

    # ------------------------------------------------------------------
    # Public API

    def grow(
        self,
        n_components: Union[float, int] = 0.95,
        min_size: int = 10,
        max_group: int = 5,
    ) -> None:
        """Grow the binary tree recursively.

        Parameters
        ----------
        n_components : float or int
            FPCA components (or variance fraction) at each node.
        min_size : int
            Minimum observations to attempt a split.
        max_group : int
            Maximum number of GMM components to evaluate with BIC.
        """
        tree = self._recursive_clustering(self.tree, n_components, min_size, max_group)
        self.tree = sorted(tree, key=lambda n: (n.depth, n.identifier
                           if isinstance(n.identifier, tuple) else (0, 0)))
        self.mapping_grow, self.labels_grow = format_label(self.get_leaves())

    def join(
        self,
        n_components: Union[float, int] = 0.95,
        max_group: int = 5,
    ) -> None:
        """Merge over-split leaves (joining step).

        Parameters
        ----------
        n_components : float or int
            FPCA components for the joint dimensionality reduction.
        max_group : int
            Maximum K in BIC search during joining.
        """
        leaves = self.get_leaves()
        siblings = self.get_siblings()
        final_clusters = self._recursive_joining(leaves, siblings, n_components, max_group)
        self.mapping_join, self.labels_join = format_label(final_clusters)

    def predict(
        self,
        new_data: Union[T, M],
        step: str = 'join',
    ) -> np.ndarray:
        """Predict cluster labels for new observations.

        Parameters
        ----------
        new_data : FunctionalData
        step : {'grow', 'join'}, default='join'

        Returns
        -------
        np.ndarray of int, shape (n_obs,)
        """
        if isinstance(new_data, DenseFunctionalData):
            return np.array([self._predict_one(obs, step) for obs in new_data])
        if isinstance(new_data, MultivariateFunctionalData):
            return np.array([self._predict_one(obs, step)
                             for obs in new_data.get_obs()])
        raise TypeError("Wrong data type.")

    # ------------------------------------------------------------------
    # Tree navigation

    def get_leaves(self) -> List[N]:
        """Return all leaf nodes."""
        return [n for n in self.tree if n.is_leaf]

    def get_siblings(self) -> Set[Tuple[N, N]]:
        """Return all sibling pairs of leaf nodes.

        Siblings share the same parent: left=(d, 2k) and right=(d, 2k+1)
        have parent identifier (d-1, k).
        """
        from collections import defaultdict
        parent_groups: Dict[tuple, List[N]] = defaultdict(list)
        for node in self.get_leaves():
            ident = node.identifier
            if isinstance(ident, tuple) and ident[0] > 0:
                parent_key = (ident[0], ident[1] // 2)
                parent_groups[parent_key].append(node)

        siblings: Set[Tuple[N, N]] = set()
        for group in parent_groups.values():
            for pair in itertools.combinations(group, 2):
                siblings.add(pair)
        return siblings

    # ------------------------------------------------------------------
    # Internal

    def _recursive_clustering(
        self,
        list_nodes: List[N],
        n_components: Union[float, int],
        min_size: int,
        max_group: int,
    ) -> List[N]:
        tree = []
        for node in list_nodes:
            if node is not None:
                tree.append(node)
                node.split(
                    n_components=n_components,
                    min_size=min_size,
                    max_group=max_group,
                )
                tree.extend(self._recursive_clustering(
                    [node.left, node.right],
                    n_components, min_size, max_group,
                ))
        return tree

    def _recursive_joining(
        self,
        list_nodes: List[N],
        siblings: Set[Tuple[N, N]],
        n_components: Union[float, int],
        max_group: int,
    ) -> List[N]:
        new_list = joining_step2(
            list_nodes, siblings, n_components, max_group,
            normalize=self.normalize,
        )
        if len(new_list) == len(list_nodes):
            return new_list
        return self._recursive_joining(new_list, siblings, n_components, max_group)

    def _map_grow_join(self) -> Dict[N, N]:
        """Map each grow-step leaf to its join-step cluster node."""
        mapping = {}
        for leaf in self.mapping_grow:
            for jnode in self.mapping_join:
                if isinstance(jnode.identifier, tuple):
                    if leaf.identifier == jnode.identifier:
                        mapping[leaf] = jnode
                elif isinstance(jnode.identifier, list):
                    if leaf.identifier in jnode.identifier:
                        mapping[leaf] = jnode
        return mapping

    def _predict_one(self, new_obs: Union[T, M], step: str = 'join') -> int:
        node = self.root_node
        while not node.is_leaf:
            node = node.predict(new_obs)

        if step == 'grow':
            return self.mapping_grow[node]
        if step == 'join':
            return self.mapping_join[self._map_grow_join()[node]]
        raise ValueError("step must be 'grow' or 'join'.")
