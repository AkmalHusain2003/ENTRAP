#!/usr/bin/env python3
"""Run ENTRAP on synthetic blobs, Two Moons, and Iris data and save a comparison plot."""
import time
import numpy as np
import matplotlib
matplotlib.use("Agg")
import matplotlib.pyplot as plt
from matplotlib.colors import LinearSegmentedColormap
from scipy.optimize import linear_sum_assignment
try:
    from sklearn.cluster import HDBSCAN as _HDBSCAN
except ImportError:  # scikit-learn >= 1.3
    try:
        from hdbscan import HDBSCAN as _HDBSCAN
    except ImportError:  # optional benchmark dependency
        _HDBSCAN = None
from sklearn.datasets import load_iris, make_moons
from sklearn.decomposition import PCA
from sklearn.metrics import adjusted_rand_score
from sklearn.preprocessing import StandardScaler

from entrap import ENTRAP, topology
from entrap.config import EntrapConfig
from entrap.leiden import has_leiden, backend_info

# ---- palet (reference palette, mode terang, 3 slot pertama tervalidasi all-pairs) ----
SURFACE, INK, INK2 = "#fcfcfb", "#0b0b0b", "#52514e"
SERIES = ["#2a78d6", "#eb6834", "#1baf7a"]          # biru, oranye, aqua
OTHER = "#52514e"                                    # klaster ke-4 dst (tidak boleh siklik)
NOISE = "#8a8984"
MARKERS = ["o", "s", "^", "D"]                       # pengkode sekunder (bentuk)
SEQ = LinearSegmentedColormap.from_list(
    "seq_blue", ["#cde2fb", "#6da7ec", "#2a78d6", "#184f95", "#0d366b"])

# P=399 gives a finer Monte-Carlo p-value grid than the bare minimum for
# five K scales and two per-scale statistics: p_adj,min = 2*M/(P+1).
# Leiden is mandatory in this demo: do not silently substitute merge-tree seeds.
ADV = {
    "B": 15,
    "P": 399,
    "pi_runs": 15,
    "stability_B": 15,
    "pi_full_stability": False,
    "seed": 0,
    # The two-moons demo is deliberately non-convex: distance-aware edge
    # filtration is required to prevent sparse bridges from merging its arcs.
    "mutual_reachability": True,
    "use_leiden": True,
    "allow_fallback": False,
    "gamma_range": (0.0, 1.0),
    "leiden_gamma_steps": 19,
    "leiden_restarts": 1,
    "leiden_iterations": -1,
}


def make_datasets():
    """Create the synthetic blobs, Two Moons, and Iris demonstration datasets."""
    rng = np.random.default_rng(0)
    centers = ([0, 0, 0], [6, 6, 0], [0, 7, 5])
    blobs = [rng.normal(c, 0.35, size=(150, 3)) for c in centers]
    noise = rng.uniform(-3, 9, size=(120, 3))
    Xa = np.vstack(blobs + [noise])
    ya = np.r_[np.repeat([0, 1, 2], 150), np.full(120, -1)]

    Xb, yb = make_moons(n_samples=400, noise=0.07, random_state=0)

    iris = load_iris()
    Xc = StandardScaler().fit_transform(iris.data)
    yc = iris.target

    return [
        ("Blob 3D + 120 titik noise seragam", Xa, ya, 20, ["blob 0", "blob 1", "blob 2"]),
        ("Two moons (non-konveks)", Xb, yb, 15, ["bulan 0", "bulan 1"]),
        ("Iris (4D, terstandar)", Xc, yc, 10, list(iris.target_names)),
    ]


def project2d(X):
    """Project the input features to two dimensions for visualization."""
    if X.shape[1] == 2:
        return X, ("x", "y")
    p = PCA(n_components=2, random_state=0)
    Z = p.fit_transform(X)
    ev = p.explained_variance_ratio_
    return Z, (f"PC1 ({ev[0]*100:.0f}%)", f"PC2 ({ev[1]*100:.0f}%)")


def align_colors(y_true, y_pred):
    """Match predicted cluster colours to reference classes for plotting."""
    clusters = [c for c in np.unique(y_pred) if c >= 0]
    classes = [c for c in np.unique(y_true) if c >= 0]
    slot = {}
    if clusters and classes:
        M = np.array([[np.sum((y_pred == c) & (y_true == t)) for t in classes] for c in clusters])
        r, c_ = linear_sum_assignment(-M)
        for ri, ci in zip(r, c_):
            if ci < len(SERIES) and M[ri, ci] > 0:
                slot[clusters[ri]] = ci
    free = [s for s in range(len(SERIES)) if s not in slot.values()]
    rest = sorted((c for c in clusters if c not in slot), key=lambda c: -np.sum(y_pred == c))
    for c in rest:
        slot[c] = free.pop(0) if free else len(SERIES)   # len(SERIES) -> "lainnya"
    return slot


def style(ax):
    """Apply the shared visual style to one Matplotlib axis."""
    ax.set_facecolor(SURFACE)
    for s in ("top", "right"):
        ax.spines[s].set_visible(False)
    for s in ("left", "bottom"):
        ax.spines[s].set_color("#c9c8c2")
    ax.tick_params(colors=INK2, labelsize=8, length=3)
    ax.grid(True, color="#ecebe6", linewidth=0.6)
    ax.set_axisbelow(True)


def scatter_labels(ax, Z, y, slot, names=None, truth=False):
    """Draw labelled cluster points and their legend entries."""
    handles = []
    if np.any(y < 0):
        m = y < 0
        ax.scatter(Z[m, 0], Z[m, 1], s=14, c=NOISE, marker="x", linewidths=0.8, zorder=2)
        handles.append(ax.scatter([], [], s=22, c=NOISE, marker="x", linewidths=0.9,
                                  label=f"noise (n={m.sum()})"))
    other_n = 0
    for c in sorted(np.unique(y[y >= 0]), key=lambda c: slot[c]):
        m = y == c
        k = slot[c]
        if k >= len(SERIES):
            other_n += 1
            col, mk = OTHER, MARKERS[3]
        else:
            col, mk = SERIES[k], MARKERS[k]
        ax.scatter(Z[m, 0], Z[m, 1], s=26, c=col, marker=mk, edgecolors=SURFACE,
                   linewidths=0.5, zorder=3)
        if truth and names is not None:
            lab = names[c] if c < len(names) else f"kelas {c}"
        else:
            lab = f"klaster {c}"
        handles.append(ax.scatter([], [], s=34, c=col, marker=mk, edgecolors=SURFACE,
                                  label=f"{lab} (n={m.sum()})"))
    return handles


def main():
    """Run the demonstration workflow and save the clustering comparison figure."""
    print("Backend Cython aktif:", topology.USING_CYTHON)
    info = backend_info()
    if not has_leiden():
        raise SystemExit(
            "Backend igraph/leidenalg tidak tersedia. Demo ini sengaja tidak "
            "mengizinkan fallback agar hasilnya tidak disalahartikan sebagai "
            "hasil Leiden. Jalankan: python -m pip install -r requirements.txt. "
            f"Detail import: {info.get('error')}"
        )
    print("Backend igraph aktif:", info["igraph_version"])
    print("Backend leidenalg aktif:", info["leidenalg_version"])
    print("Partition objective: CPMVertexPartition")
    if _HDBSCAN is None:
        print(
            "Peringatan: baseline HDBSCAN tidak tersedia. "
            "Install hdbscan atau scikit-learn>=1.3."
        )
    data = make_datasets()
    fig, axes = plt.subplots(len(data), 3, figsize=(15.5, 13.8), facecolor=SURFACE)
    rows = []

    for i, (title, X, y, mcs, names) in enumerate(data):
        # Check that the empirical Monte-Carlo resolution can reach the
        # family-wise threshold for the actual candidate scale count.
        resolved = EntrapConfig.from_params(mcs, 0.05, ADV).resolve(len(X))
        n_scales = len(resolved.k_scales_resolved)
        p_adjusted_floor = 2.0 * n_scales / (resolved.P + 1.0)
        if p_adjusted_floor > 0.05 + 1e-12:
            min_p = int(np.ceil(2.0 * n_scales / 0.05) - 1)
            raise RuntimeError(
                f"P={resolved.P} cannot resolve alpha=0.05 after the "
                f"2 x {n_scales} Bonferroni tests, require P >= {min_p}."
            )

        t0 = time.time()
        model = ENTRAP(min_cluster_size=mcs, significance=0.05, advanced=ADV).fit(X)
        dt = time.time() - t0
        lab = model.labels_
        pi = model.result_.probabilities
        if _HDBSCAN is None:
            hb = np.full(len(X), -1, dtype=np.int64)
            ari_h = np.nan
            hdbscan_count = -1
        else:
            hb = _HDBSCAN(min_cluster_size=mcs).fit(X).labels_
            ari_h = adjusted_rand_score(y, hb)
            hdbscan_count = len(np.unique(hb[hb >= 0]))

        ari_e = adjusted_rand_score(y, lab)
        root = model.result_.tree
        root_adj_p = root.k_adjusted_p
        ld = model.leiden_diagnostics_
        rows.append((title, len(X), X.shape[1], len(np.unique(y[y >= 0])),
                     int(model.result_.n_clusters), int(np.sum(lab < 0)),
                     model.result_.noise_rescued, ari_e,
                     hdbscan_count, ari_h, dt,
                     int(ld["calls"]), int(ld["attempted"]),
                     int(ld["optimised"]), int(ld["seed_fallbacks"]), root_adj_p))
        print(
            f"[{title}] root: m_star={root.m_star}, "
            f"adjusted_p={root_adj_p}, selected_K={root.selected_k}, "
            f"Leiden calls={ld['calls']}, attempted={ld['attempted']}, "
            f"optimised={ld['optimised']}, seed_fallbacks={ld['seed_fallbacks']}"
        )
        if root.stability_selected_k is not None:
            print(
                "  Multi-K stability: selected K per subsample="
                f"{root.stability_selected_k.tolist()}, "
                f"mode counts={root.stability_mode_counts.tolist()}"
            )
        elif root.stability_skipped_reason:
            print("  Multi-K stability skipped:", root.stability_skipped_reason)
        if ld["calls"] == 0:
            print(
                "  Diagnosis: Leiden was not invoked because the root did not "
                "request a topological split (m_star <= 1). Inspect root "
                "persistence/significance diagnostics before tuning CPM gamma."
            )
        for item in ld["nodes"]:
            if item.get("status") == "seed_fallback_no_exact_target_partition":
                print(
                    f"  Leiden ran at depth={item['depth']} (n={item['node_size']}) "
                    f"but no scanned gamma retained target={item['target_communities']}, "
                    "the topological seed was retained. Inspect node diagnostics "
                    "or widen gamma_range/increase leiden_gamma_steps."
                )

        Z, (xl, yl) = project2d(X)
        slot_true = {c: j for j, c in enumerate(sorted(np.unique(y[y >= 0])))}
        slot_pred = align_colors(y, lab)

        ax0, ax1, ax2 = axes[i]
        for ax in (ax0, ax1, ax2):
            style(ax)
            ax.set_xlabel(xl, color=INK2, fontsize=9)
        ax0.set_ylabel(yl, color=INK2, fontsize=9)

        h0 = scatter_labels(ax0, Z, y, slot_true, names=names, truth=True)
        ax0.set_title("Label sebenarnya", loc="left", fontsize=10.5, color=INK, pad=6)
        ax0.legend(handles=h0, loc="upper center", bbox_to_anchor=(0.5, -0.17),
                   ncol=min(len(h0), 4), frameon=False, fontsize=8, labelcolor=INK2,
                   columnspacing=1.0, handletextpad=0.3)

        h1 = scatter_labels(ax1, Z, lab, slot_pred)
        ax1.set_title(f"ENTRAP: {model.result_.n_clusters} klaster, ARI = {ari_e:.2f}",
                      loc="left", fontsize=10.5, color=INK, pad=6)
        ax1.legend(handles=h1, loc="upper center", bbox_to_anchor=(0.5, -0.17),
                   ncol=min(len(h1), 4), frameon=False, fontsize=8, labelcolor=INK2,
                   columnspacing=1.0, handletextpad=0.3)

        order = np.argsort(pi)
        sc = ax2.scatter(Z[order, 0], Z[order, 1], c=pi[order], cmap=SEQ, vmin=0, vmax=1,
                         s=26, edgecolors=SURFACE, linewidths=0.4)
        ax2.set_title("Probabilitas keanggotaan (stabilitas subsampel)",
                      loc="left", fontsize=10.5, color=INK, pad=6)
        cb = fig.colorbar(sc, ax=ax2, fraction=0.046, pad=0.03)
        cb.outline.set_visible(False)
        cb.ax.tick_params(colors=INK2, labelsize=8)

        fig.text(0.012, 1 - (i + 0.02) / len(data) * 0.93 - 0.035, title, fontsize=12,
                 fontweight="bold", color=INK, ha="left", va="top")

    fig.suptitle("ENTRAP pada data sintetis dan Iris", x=0.012, y=0.995, ha="left",
                 fontsize=15, fontweight="bold", color=INK)
    fig.text(
        0.012,
        0.012,
        "Warna klaster ENTRAP dicocokkan ke kelas sebenarnya "
        "(Hungarian) hanya untuk keterbacaan. "
        f"Pengaturan: B={ADV['B']}, P={ADV['P']}, pi_runs={ADV['pi_runs']}, "
        f"CPM gamma steps={ADV['leiden_gamma_steps']}, significance=0.05, seed=0. "
        "mutual_reachability=True, multi-K stability memilih ulang K "
        "pada setiap subsample, Leiden wajib aktif, Iris diproyeksikan ke 2 PC.",
        fontsize=8.5,
        color=INK2,
        ha="left",
    )
    fig.subplots_adjust(left=0.05, right=0.985, top=0.925, bottom=0.09, hspace=0.62, wspace=0.22)
    fig.savefig("demo_clustering.png", dpi=150, facecolor=SURFACE)

    hdr = ("dataset", "N", "d", "k_benar", "k_ENTRAP", "noise", "rescued",
           "ARI_ENTRAP", "k_HDBSCAN", "ARI_HDBSCAN", "detik",
           "Leiden_calls", "Leiden_attempted", "Leiden_optimised",
           "Leiden_seed_fallbacks", "root_adjusted_p")
    print("\n" + " | ".join(hdr))
    for r in rows:
        root_p = "None" if r[15] is None else f"{r[15]:.4g}"
        print(f"{r[0]} | {r[1]} | {r[2]} | {r[3]} | {r[4]} | {r[5]} | {r[6]} | "
              f"{r[7]:.3f} | {r[8]} | {r[9]:.3f} | {r[10]:.1f} | "
              f"{r[11]} | {r[12]} | {r[13]} | {r[14]} | {root_p}")
    print("\nTersimpan: demo_clustering.png")


if __name__ == "__main__":
    main()
