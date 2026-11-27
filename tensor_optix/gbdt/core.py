"""
Histogram gradient-boosted decision trees in numpy.

* Features are binned once (quantile bins for numeric features, frequency-ranked
  codes for categorical ones); each feature has its own *missing* bin.
* Trees grow depth-wise. At every level the gradient / hessian / count
  histograms of all active nodes are built with one ``bincount`` per feature,
  and the best split of every node is found with cumulative sums over bins.
* Split gain is the second-order (Newton) gain with L2 regularisation:
  ``G_L^2 / (H_L + lambda) + G_R^2 / (H_R + lambda) - G^2 / (H + lambda) - gamma``.
* Missing values: each split tries sending the missing bin left and right and
  keeps the better direction (also used for unseen values at predict time).
* Categorical features: within a node, categories are sorted by G / (H + lambda)
  and the best prefix of that order goes left (the optimal binary partition
  for a convex loss, as in LightGBM / Fisher 1958).
"""
from __future__ import annotations

from dataclasses import dataclass, field
from typing import List, Optional, Sequence

import numpy as np


# ---------------------------------------------------------------------------
# binning
# ---------------------------------------------------------------------------

@dataclass
class BinMapper:
    max_bins: int = 255
    categorical: Sequence[int] = ()
    max_categories: int = 254
    thresholds: List[Optional[np.ndarray]] = field(default_factory=list)   # numeric: bin upper edges
    categories: List[Optional[np.ndarray]] = field(default_factory=list)   # categorical: kept values
    n_bins: List[int] = field(default_factory=list)                         # incl. the missing bin

    def fit(self, X: np.ndarray) -> "BinMapper":
        X = np.asarray(X, dtype=np.float64)
        cat = set(int(c) for c in self.categorical)
        self.thresholds, self.categories, self.n_bins = [], [], []
        for j in range(X.shape[1]):
            x = X[:, j]
            ok = np.isfinite(x)
            if j in cat:
                vals, cnt = np.unique(x[ok], return_counts=True)
                keep = vals[np.argsort(-cnt, kind="stable")][: self.max_categories]
                self.categories.append(np.sort(keep))
                self.thresholds.append(None)
                self.n_bins.append(len(keep) + 2)                  # + "other" + missing
                continue
            u = np.unique(x[ok])
            if len(u) <= 1:
                edges = np.array([], dtype=np.float64)
            elif len(u) <= self.max_bins - 1:
                edges = (u[:-1] + u[1:]) / 2.0
            else:
                q = np.quantile(x[ok], np.linspace(0, 1, self.max_bins)[1:-1])
                edges = np.unique(q)
            self.thresholds.append(edges)
            self.categories.append(None)
            self.n_bins.append(len(edges) + 2)                      # bins 0..len(edges), + missing
        return self

    def transform(self, X: np.ndarray) -> np.ndarray:
        X = np.asarray(X, dtype=np.float64)
        dtype = np.uint8 if max(self.n_bins, default=2) <= 256 else np.uint16
        out = np.empty(X.shape, dtype=dtype)
        for j in range(X.shape[1]):
            x = X[:, j]
            miss = ~np.isfinite(x)
            if self.categories[j] is not None:
                cats = self.categories[j]
                pos = np.searchsorted(cats, x)
                pos_c = np.clip(pos, 0, max(len(cats) - 1, 0))
                found = (len(cats) > 0) & (cats[pos_c] == x) if len(cats) else np.zeros(len(x), dtype=bool)
                b = np.where(found, pos_c, len(cats))               # unseen -> "other"
            else:
                b = np.searchsorted(self.thresholds[j], np.where(miss, 0.0, x), side="left")
            b = np.where(miss, self.n_bins[j] - 1, b)
            out[:, j] = b
        return out

    def missing_bin(self, j: int) -> int:
        return self.n_bins[j] - 1


# ---------------------------------------------------------------------------
# a single tree
# ---------------------------------------------------------------------------

@dataclass
class Tree:
    feature: np.ndarray          # -1 for leaves
    threshold: np.ndarray        # numeric: go left when bin <= threshold
    missing_left: np.ndarray
    cat_index: np.ndarray        # -1 numeric; else row of cat_left
    cat_left: np.ndarray         # (n_cat_nodes, max_bins) bool: categorical bin goes left
    left: np.ndarray
    right: np.ndarray
    value: np.ndarray            # leaf values (already scaled by the learning rate)
    depth: int

    def predict_binned(self, Xb: np.ndarray, missing_bins: np.ndarray) -> np.ndarray:
        n = len(Xb)
        node = np.zeros(n, dtype=np.int64)
        rows = np.arange(n)
        for _ in range(self.depth):
            f = self.feature[node]
            internal = f >= 0
            if not internal.any():
                break
            r = rows[internal]
            nd = node[internal]
            ff = f[internal]
            b = Xb[r, ff].astype(np.int64)
            is_miss = b == missing_bins[ff]
            ci = self.cat_index[nd]
            go_left = np.where(is_miss, self.missing_left[nd], b <= self.threshold[nd])
            cat_rows = ci >= 0
            if cat_rows.any():
                cb = np.minimum(b[cat_rows], self.cat_left.shape[1] - 1)
                go_left[cat_rows] = np.where(is_miss[cat_rows], self.missing_left[nd[cat_rows]],
                                             self.cat_left[ci[cat_rows], cb])
            node[internal] = np.where(go_left, self.left[nd], self.right[nd])
        return self.value[node]


@dataclass
class TreeParams:
    max_depth: int = 6
    min_samples_leaf: int = 20
    min_child_weight: float = 1e-3
    reg_lambda: float = 1.0
    gamma: float = 0.0
    learning_rate: float = 0.1
    # categorical regularisation, as in LightGBM: categories with fewer than ``cat_smooth`` rows do not
    # take part in the partition search (they stay right) and the ratio that orders categories is
    # smoothed by it; categorical split gains get an extra ``cat_l2``; each side of a categorical split
    # needs ``min_data_per_group`` rows; at most ``max_cat_threshold`` categories go to one side
    cat_smooth: float = 10.0
    cat_l2: float = 10.0
    min_data_per_group: int = 100
    max_cat_threshold: int = 32


def build_tree(Xb: np.ndarray, g: np.ndarray, h: np.ndarray, n_bins: Sequence[int], is_cat: np.ndarray,
               features: np.ndarray, p: TreeParams) -> Tree:
    """Grow one tree depth-wise on binned data (rows already subsampled by the caller)."""
    n = len(g)
    max_nodes = 2 ** (p.max_depth + 1) - 1
    B = int(max(n_bins))
    feature = np.full(max_nodes, -1, dtype=np.int64)
    threshold = np.zeros(max_nodes, dtype=np.int64)
    missing_left = np.zeros(max_nodes, dtype=bool)
    cat_index = np.full(max_nodes, -1, dtype=np.int64)
    left = np.zeros(max_nodes, dtype=np.int64)
    right = np.zeros(max_nodes, dtype=np.int64)
    value = np.zeros(max_nodes, dtype=np.float64)
    cat_rows: List[np.ndarray] = []
    lam = p.reg_lambda

    node_of = np.zeros(n, dtype=np.int64)          # position among this level's active nodes
    active = [0]                                    # global ids of this level's nodes
    G_node = np.array([g.sum()])
    H_node = np.array([h.sum()])
    C_node = np.array([n])
    next_id = 1
    depth_used = 0
    for depth in range(p.max_depth):
        k = len(active)
        best_gain = np.full(k, 0.0)
        best_f = np.full(k, -1, dtype=np.int64)
        best_thr = np.zeros(k, dtype=np.int64)
        best_ml = np.zeros(k, dtype=bool)
        best_cat: List[Optional[np.ndarray]] = [None] * k
        can_split = C_node >= 2 * p.min_samples_leaf
        if not can_split.any():
            break
        for f in features:
            nb = int(n_bins[f])
            miss = nb - 1
            idx = node_of * nb + Xb[:, f].astype(np.int64)
            Gh = np.bincount(idx, weights=g, minlength=k * nb).reshape(k, nb)
            Hh = np.bincount(idx, weights=h, minlength=k * nb).reshape(k, nb)
            Ch = np.bincount(idx, minlength=k * nb).reshape(k, nb)
            Gm, Hm, Cm = Gh[:, miss], Hh[:, miss], Ch[:, miss]
            Gv, Hv, Cv = Gh[:, :miss], Hh[:, :miss], Ch[:, :miss]
            order = None
            lam_f = lam
            if is_cat[f]:
                lam_f = lam + p.cat_l2
                ratio = Gv / (Hv + p.cat_smooth)
                ratio = np.where(Cv >= p.cat_smooth, ratio, np.inf)     # rare / empty categories last
                order = np.argsort(ratio, axis=1, kind="stable")
                Gv = np.take_along_axis(Gv, order, 1)
                Hv = np.take_along_axis(Hv, order, 1)
                Cv = np.take_along_axis(Cv, order, 1)
            GL = np.cumsum(Gv, axis=1)[:, :-1]
            HL = np.cumsum(Hv, axis=1)[:, :-1]
            CL = np.cumsum(Cv, axis=1)[:, :-1]
            if GL.shape[1] == 0:
                continue
            Gt, Ht, Ct = G_node[:, None], H_node[:, None], C_node[:, None]
            pscore = (G_node ** 2 / (H_node + lam_f))[:, None]
            if is_cat[f]:
                # a prefix may only end at a category that took part in the search
                n_elig = (Cv >= p.cat_smooth).sum(axis=1, keepdims=True)     # Cv is in sorted order
                plen = np.arange(1, Cv.shape[1])[None, :]                  # categories sent left
                # a prefix of participating categories, with at most max_cat_threshold on one side
                eligible = (plen <= n_elig) & ((plen <= p.max_cat_threshold) |
                                               (n_elig - plen <= p.max_cat_threshold))
            gains = []
            for ml in (False, True):
                gl = GL + (Gm[:, None] if ml else 0.0)
                hl = HL + (Hm[:, None] if ml else 0.0)
                cl = CL + (Cm[:, None] if ml else 0)
                gr, hr, cr = Gt - gl, Ht - hl, Ct - cl
                gain = gl ** 2 / (hl + lam_f) + gr ** 2 / (hr + lam_f) - pscore - p.gamma
                bad = ((cl < p.min_samples_leaf) | (cr < p.min_samples_leaf) |
                       (hl < p.min_child_weight) | (hr < p.min_child_weight))
                if is_cat[f]:
                    bad = bad | ~eligible | (cl < p.min_data_per_group) | (cr < p.min_data_per_group)
                gains.append(np.where(bad, -np.inf, gain))
            g0, g1 = gains
            use_ml = g1 > g0
            gbest = np.where(use_ml, g1, g0)
            pos = np.argmax(gbest, axis=1)
            val = gbest[np.arange(k), pos]
            better = (val > best_gain) & can_split
            for i in np.flatnonzero(better):
                best_gain[i] = val[i]
                best_f[i] = f
                best_ml[i] = bool(use_ml[i, pos[i]])
                if is_cat[f]:
                    mask = np.zeros(B, dtype=bool)
                    mask[order[i, : pos[i] + 1]] = True
                    best_cat[i] = mask
                    best_thr[i] = -1
                else:
                    best_thr[i] = pos[i]
                    best_cat[i] = None
        split = best_f >= 0
        if not split.any():
            break
        depth_used = depth + 1
        new_active, nG, nH, nC = [], [], [], []
        child_pos = np.full((k, 2), -1, dtype=np.int64)
        for i in range(k):
            gid = active[i]
            if not split[i]:
                continue
            feature[gid] = best_f[i]
            threshold[gid] = best_thr[i]
            missing_left[gid] = best_ml[i]
            if best_cat[i] is not None:
                cat_index[gid] = len(cat_rows)
                cat_rows.append(best_cat[i])
            left[gid], right[gid] = next_id, next_id + 1
            child_pos[i] = (len(new_active), len(new_active) + 1)
            new_active += [next_id, next_id + 1]
            next_id += 2
        # route rows
        f_row = best_f[node_of]
        moving = f_row >= 0
        b = Xb[np.flatnonzero(moving), f_row[moving]].astype(np.int64)
        nd = node_of[moving]
        miss_b = np.asarray(n_bins, dtype=np.int64)[f_row[moving]] - 1
        is_miss = b == miss_b
        go_left = np.where(is_miss, best_ml[nd], b <= best_thr[nd])
        cat_nodes = [i for i in range(k) if best_cat[i] is not None]
        if cat_nodes:
            table = np.zeros((k, B), dtype=bool)
            has_cat = np.zeros(k, dtype=bool)
            for i in cat_nodes:
                table[i] = best_cat[i]
                has_cat[i] = True
            sel = has_cat[nd]
            go_left[sel] = np.where(is_miss[sel], best_ml[nd[sel]], table[nd[sel], np.minimum(b[sel], B - 1)])
        new_pos = np.where(go_left, child_pos[nd, 0], child_pos[nd, 1])
        # leaves of this level keep their value; stop rows that reached a leaf
        node_after = np.full(n, -1, dtype=np.int64)
        node_after[np.flatnonzero(moving)] = new_pos
        # finalise leaves that did not split
        for i in range(k):
            if not split[i]:
                value[active[i]] = -G_node[i] / (H_node[i] + lam) * p.learning_rate
        kk = len(new_active)
        rows_alive = node_after >= 0
        nG = np.bincount(node_after[rows_alive], weights=g[rows_alive], minlength=kk)
        nH = np.bincount(node_after[rows_alive], weights=h[rows_alive], minlength=kk)
        nC = np.bincount(node_after[rows_alive], minlength=kk)
        # rows that stopped are dropped from further histogram work
        keep = rows_alive
        Xb, g, h = Xb[keep], g[keep], h[keep]
        node_of = node_after[keep]
        n = len(g)
        active, G_node, H_node, C_node = new_active, nG, nH, nC
        if n == 0:
            break
    for i, gid in enumerate(active):
        if feature[gid] < 0:
            value[gid] = -G_node[i] / (H_node[i] + lam) * p.learning_rate
    m = next_id
    cat_left = np.array(cat_rows, dtype=bool) if cat_rows else np.zeros((0, B), dtype=bool)
    return Tree(feature=feature[:m], threshold=threshold[:m], missing_left=missing_left[:m],
                cat_index=cat_index[:m], cat_left=cat_left, left=left[:m], right=right[:m], value=value[:m],
                depth=max(depth_used, 1))
