#!/usr/bin/env python
# -*-coding:utf8 -*

"""
Multi-branch functional CUBT (mCUBT).

Extends fCUBT (FDApy) by allowing each node to split into K >= 2 children,
where K is the number of Gaussian components selected by BIC — instead of
always forcing a binary (K=2) split.

Key differences from fCUBT
---------------------------
1. Growing step  : the node splits into K̂ children using GMM(K̂), not GMM(2).
2. Node identity : path-based identifier (tuple of branch indices from root)
                   instead of the binary (depth, position) scheme.
3. Children list : MNode.children replaces the binary left/right attributes.
4. Routing       : MNode.predict() routes to argmax of K̂ GMM responsibilities.
5. Joining step  : sibling pairs are all pairs sharing the same parent path
                   (instead of just binary left-right pairs).

Usage
-----
    from mcubt import MNode, MCUBT

    root = MNode(data_fd, is_root=True)
    mcubt = MCUBT(root_node=root)
    mcubt.grow(n_components=0.95, min_size=10, max_group=8, min_group_size=10)
    mcubt.join(n_components=0.95, max_group=8)
    labels = mcubt.labels_join
"""

import itertools
from collections import defaultdict
from typing import Dict, List, Optional, Set, Tuple, TypeVar, Union

import networkx as nx
import numpy as np
from sklearn.mixture import GaussianMixture

from FDApy.clustering.fcubt import format_label, joining_step
from FDApy.clustering.optimalK.bic import BIC
from FDApy.preprocessing.dim_reduction.fcp_tpa import FCPTPA
from FDApy.preprocessing.dim_reduction.fpca import MFPCA, UFPCA
from FDApy.representation.functional_data import (DenseFunctionalData,
                                                   MultivariateFunctionalData)

N = TypeVar('N', bound='MNode')
T = TypeVar('T', bound='DenseFunctionalData')
M = TypeVar('M', bound='MultivariateFunctionalData')


###############################################################################
# MNode

class MNode:
    """A node in the mCUBT multi-branch tree.

    Parameters
    ----------
    data : DenseFunctionalData or MultivariateFunctionalData
        Functional data held by this node.
    path : tuple or list, default=()
        Unique path identifier from the root.
        - Root node  : ()
        - k-th child (0-indexed) of node with path p : p + (k,)
        - Merged node (after join step) : list of constituent paths.
    idx_obs : np.ndarray, default=None
        Global observation indices tracked through splits.
    is_root : bool, default=False
    is_leaf  : bool, default=False
    normalize : bool, default=False
        Normalize data before FPCA.

    Attributes
    ----------
    children : list of MNode
        Child nodes created during the split.
    labels : np.ndarray
        GMM component assignment for each observation in this node.
    fpca : FPCA model
        Fitted dimensionality-reduction model.
    gaussian_model : GaussianMixture
        Fitted GMM with K̂ components.
    best_k : int
        Number of children chosen by BIC.
    """

    def __init__(
        self,
        data: Union[T, M],
        path: Union[tuple, list] = (),
        idx_obs: Optional[np.ndarray] = None,
        is_root: bool = False,
        is_leaf: bool = False,
        normalize: bool = False,
    ) -> None:
        if not isinstance(data, (DenseFunctionalData, MultivariateFunctionalData)):
            raise TypeError("data must be DenseFunctionalData or MultivariateFunctionalData.")
        self.data = data
        self.path = () if is_root else path
        self.idx_obs = np.arange(data.n_obs) if idx_obs is None else idx_obs
        self.labels = np.zeros(data.n_obs, dtype=int)
        self.children: List[N] = []
        self.is_root = is_root
        self.is_leaf = is_leaf
        self.normalize = normalize
        self.fpca = None
        self.gaussian_model = None
        self.best_k = None

    # ------------------------------------------------------------------
    # Properties

    @property
    def depth(self) -> int:
        """Depth of this node (root = 0)."""
        p = self.path
        if isinstance(p, tuple):
            return len(p)
        if isinstance(p, list) and p:
            first = p[0]
            return len(first) if isinstance(first, tuple) else 0
        return 0

    # ------------------------------------------------------------------
    # Repr

    def __str__(self) -> str:
        return (f"MNode(path={self.path}, depth={self.depth}, "
                f"n_obs={self.data.n_obs}, is_root={self.is_root}, "
                f"is_leaf={self.is_leaf})")

    def __repr__(self) -> str:
        return self.__str__()

    # ------------------------------------------------------------------
    # Internal helpers

    def _compute_scores(
        self,
        n_components: Union[float, int]
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
        min_group_size: int = 10,
    ) -> None:
        """Split this node into K̂ children selected by BIC.

        Parameters
        ----------
        n_components : float or int, default=0.95
            Number of FPCA components (or variance fraction) to retain.
        min_size : int, default=10
            Minimum observations required to attempt a split.
        max_group : int, default=5
            Maximum K to evaluate in BIC search.
        min_group_size : int, default=10
            Back-off guard: reduce K until every child has at least this
            many observations.
        """
        if self.data.n_obs <= min_size:
            self.is_leaf = True
            return

        scores, fpca = self._compute_scores(n_components)
        self.fpca = fpca

        # --- BIC selection of optimal K ---
        max_k = min(max_group, self.data.n_obs)
        bic_stat = BIC(parallel_backend=None)
        best_k = bic_stat(scores, np.arange(1, max_k))

        if best_k <= 1:
            self.is_leaf = True
            return

        # --- Back-off: reduce K until all children are large enough ---
        gm = None
        prediction = None
        while best_k > 1:
            gm = GaussianMixture(n_components=best_k, random_state=0)
            prediction = gm.fit_predict(scores)
            group_sizes = [np.sum(prediction == k) for k in range(best_k)]
            if min(group_sizes) >= min_group_size:
                break
            best_k -= 1

        if best_k <= 1:
            self.is_leaf = True
            return

        # --- Store split results ---
        self.gaussian_model = gm
        self.best_k = best_k
        self.labels = prediction

        # --- Create children ---
        for k in range(best_k):
            mask = prediction == k
            child = MNode(
                data=self._subset_data(mask),
                path=self.path + (k,),
                idx_obs=self.idx_obs[mask],
                normalize=self.normalize,
            )
            self.children.append(child)

    def unite(self, node: N) -> N:
        """Merge this node with `node` into a single MNode.

        The resulting node's path is a list of the two constituent paths,
        mirroring the identifier list convention used in fCUBT.

        Parameters
        ----------
        node : MNode

        Returns
        -------
        MNode
        """
        data = self.data.concatenate(node.data)

        # Flatten paths into a list (supports chained merges)
        self_paths = self.path if isinstance(self.path, list) else [self.path]
        node_paths = node.path if isinstance(node.path, list) else [node.path]
        new_path = self_paths + node_paths

        return MNode(
            data=data,
            path=new_path,
            idx_obs=np.hstack([self.idx_obs, node.idx_obs]),
            is_root=(self.is_root and node.is_root),
            is_leaf=(self.is_leaf and node.is_leaf),
            normalize=self.normalize,
        )

    def predict(self, new_obs: Union[T, M]) -> N:
        """Route a new observation to its best child (argmax GMM responsibility).

        Parameters
        ----------
        new_obs : FunctionalData
            A single new observation (wrapping a single curve / multivariate fd).

        Returns
        -------
        MNode  — the selected child node.
        """
        score = self.fpca.transform(new_obs, method='NumInt')
        pred = int(self.gaussian_model.predict(score)[0])
        return self.children[pred]

    def predict_proba(self, new_obs: Union[T, M]) -> np.ndarray:
        """Return GMM component probabilities for a new observation.

        Returns
        -------
        np.ndarray of shape (1, K̂)
        """
        score = self.fpca.transform(new_obs, method='NumInt')
        return self.gaussian_model.predict_proba(score)


###############################################################################
# Joining step (adapted for MNode; logic identical to fCUBT joining_step)

def joining_step_m(
    list_nodes: List[N],
    siblings: Set[Tuple[N, N]],
    n_components: Union[int, float] = 0.95,
    max_group: int = 5,
    normalize: bool = False,
) -> List[N]:
    """One round of the mCUBT joining step.

    Tests all non-sibling leaf pairs. Merges the pair with the smallest BIC
    among those whose union is best described by a single Gaussian (K̂ = 1).

    Parameters
    ----------
    list_nodes : list of MNode
        Current set of leaf nodes.
    siblings : set of (MNode, MNode)
        Sibling pairs to exclude from merging candidates.
    n_components : float or int
        FPCA components for the joint dimensionality reduction.
    max_group : int
        Maximum K in BIC search.
    normalize : bool
        Normalize data before FPCA.

    Returns
    -------
    list of MNode — updated set of leaf nodes after (at most) one merge.
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
# MCUBT

class MCUBT:
    """Multi-branch functional CUBT.

    Parameters
    ----------
    root_node : MNode
        Root node of the tree (wraps the full dataset).
    normalize : bool, default=False
        Normalize data before FPCA at each node.

    Attributes
    ----------
    tree : list of MNode
        All nodes (internal + leaves) in BFS order.
    labels_grow : np.ndarray
        Cluster labels after the growing step.
    labels_join : np.ndarray
        Cluster labels after the joining step.
    mapping_grow, mapping_join : dict
        Mapping from leaf MNode → integer cluster label.
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
        min_group_size: int = 10,
    ) -> None:
        """Grow the multi-branch tree recursively.

        Parameters
        ----------
        n_components : float or int
            FPCA components (or variance fraction) at each node.
        min_size : int
            Minimum observations to attempt a split.
        max_group : int
            Maximum number of GMM components to evaluate.
        min_group_size : int
            Minimum child size for a valid split (back-off guard).
        """
        tree = self._recursive_clustering(
            self.tree, n_components, min_size, max_group, min_group_size
        )
        self.tree = sorted(tree, key=self._sort_key)
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
        final_cluster = self._recursive_joining(
            leaves, siblings, n_components, max_group
        )
        self.mapping_join, self.labels_join = format_label(final_cluster)

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
            return np.array([self._predict(obs, step) for obs in new_data])
        if isinstance(new_data, MultivariateFunctionalData):
            return np.array([self._predict(obs, step)
                             for obs in new_data.get_obs()])
        raise TypeError("Wrong data type.")

    def predict_proba(
        self,
        new_data: Union[T, M],
        step: str = 'join',
    ) -> list:
        """Predict cluster probabilities for new observations.

        Parameters
        ----------
        new_data : FunctionalData
        step : {'grow', 'join'}, default='join'

        Returns
        -------
        list of dict  (one dict per observation, mapping leaf MNode → float)
        """
        if isinstance(new_data, DenseFunctionalData):
            return [self._predict_proba(obs, step) for obs in new_data]
        if isinstance(new_data, MultivariateFunctionalData):
            return [self._predict_proba(obs, step)
                    for obs in new_data.get_obs()]
        raise TypeError("Wrong data type.")

    # ------------------------------------------------------------------
    # Tree navigation

    def get_leaves(self) -> List[N]:
        """Return all leaf nodes."""
        return [n for n in self.tree if n.is_leaf]

    def get_siblings(self) -> Set[Tuple[N, N]]:
        """Return all pairs of sibling leaf nodes (same parent path).

        In mCUBT a K-way split produces K siblings; all C(K,2) pairs are
        returned so they are excluded from the joining step candidates.
        """
        parent_groups: Dict[tuple, List[N]] = defaultdict(list)
        for node in self.get_leaves():
            if isinstance(node.path, tuple) and len(node.path) > 0:
                parent_groups[node.path[:-1]].append(node)

        siblings: Set[Tuple[N, N]] = set()
        for group in parent_groups.values():
            for pair in itertools.combinations(group, 2):
                siblings.add(pair)
        return siblings

    def get_parent(self, node: N) -> Optional[N]:
        """Return the parent of `node` (None for root or merged nodes)."""
        if not isinstance(node.path, tuple) or len(node.path) == 0:
            return None
        parent_path = node.path[:-1]
        for n in self.tree:
            if isinstance(n.path, tuple) and n.path == parent_path:
                return n
        return None

    def _get_node_by_path(self, path: tuple) -> Optional[N]:
        for n in self.tree:
            if isinstance(n.path, tuple) and n.path == path:
                return n
        return None

    # ------------------------------------------------------------------
    # Internal

    @staticmethod
    def _sort_key(node: N) -> Tuple:
        p = node.path
        if isinstance(p, tuple):
            return (len(p), p)
        if isinstance(p, list) and p:
            first = p[0] if isinstance(p[0], tuple) else ()
            return (len(first), first)
        return (0, ())

    def _recursive_clustering(
        self,
        list_nodes: List[N],
        n_components: Union[float, int],
        min_size: int,
        max_group: int,
        min_group_size: int,
    ) -> List[N]:
        tree = []
        for node in list_nodes:
            if node is not None:
                tree.append(node)
                node.split(
                    n_components=n_components,
                    min_size=min_size,
                    max_group=max_group,
                    min_group_size=min_group_size,
                )
                tree.extend(self._recursive_clustering(
                    node.children,
                    n_components, min_size, max_group, min_group_size,
                ))
        return tree

    def _recursive_joining(
        self,
        list_nodes: List[N],
        siblings: Set[Tuple[N, N]],
        n_components: Union[float, int],
        max_group: int,
    ) -> List[N]:
        new_list = joining_step_m(
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
                if isinstance(jnode.path, tuple):
                    if leaf.path == jnode.path:
                        mapping[leaf] = jnode
                elif isinstance(jnode.path, list):
                    if leaf.path in jnode.path:
                        mapping[leaf] = jnode
        return mapping

    def _predict(self, new_data: Union[T, M], step: str = 'join') -> int:
        node = self.root_node
        while not node.is_leaf:
            node = node.predict(new_data)

        if step == 'grow':
            return self.mapping_grow[node]
        if step == 'join':
            return self.mapping_join[self._map_grow_join()[node]]
        raise ValueError("step must be 'grow' or 'join'.")

    def _predict_proba(
        self,
        new_data: Union[T, M],
        step: str = 'join',
    ) -> dict:
        # --- Conditional probability of each node given its parent ---
        proba_cond: Dict[N, float] = {self.root_node: 1.0}
        for node in self.tree:
            if not node.is_leaf and node.children:
                pred = node.predict_proba(new_data)[0]   # shape (K̂,)
                for k, child in enumerate(node.children):
                    proba_cond[child] = float(pred[k])

        # --- Marginal probability of each leaf (product along path) ---
        proba_grow: Dict[N, float] = {}
        for leaf in self.get_leaves():
            proba = proba_cond.get(leaf, 0.0)
            parent = self.get_parent(leaf)
            while parent is not None:
                proba *= proba_cond.get(parent, 1.0)
                parent = self.get_parent(parent)
            proba_grow[leaf] = proba

        if step == 'grow':
            return proba_grow

        if step == 'join':
            proba_join: Dict[N, float] = {}
            for jnode in self.mapping_join:
                total = 0.0
                if isinstance(jnode.path, tuple):
                    total = proba_grow.get(jnode, 0.0)
                elif isinstance(jnode.path, list):
                    for p in jnode.path:
                        n = self._get_node_by_path(p)
                        if n is not None:
                            total += proba_grow.get(n, 0.0)
                proba_join[jnode] = total
            return proba_join

        raise ValueError("step must be 'grow' or 'join'.")
