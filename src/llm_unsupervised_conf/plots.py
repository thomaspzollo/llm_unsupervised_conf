import pandas as pd
from sklearn.calibration import calibration_curve
import numpy as np

import matplotlib.pyplot as plt

plt.style.use('seaborn-v0_8')
pal = plt.rcParams['axes.prop_cycle'].by_key()['color']


METHODS_MAP = {
    "verbal_conf": "Verbal Conf.",
    "vc": "Verbal Conf.",
    "ridge_clip": "Ours",
    "split_isotonic_on_ridge": "Ours",
    "split_isotonic_on_ridge_nrt": "Ours",
    "ours": "Ours",
    "logprob": "Token Probs.",
    "lp": "Token Probs.",
    "ans_logprob": "Ans. Probs.",
    "alp": "Ans. Probs.",
    "oracle_sc": "Self. cons.",
    "ablation": "Ours (Ablation)"
}

METHODS_MAP_SQUEEZE = {
    "verbal_conf": "Verbal\nConf.",
    "ridge_clip": "Ours",
    "split_isotonic_on_ridge": "Ours",
    "split_isotonic_on_ridge_nrt": "Ours",
    "logprob": "Token\nProbs",
    "ans_logprob": "Ans.\nProbs",
    "oracle_sc": "Self.\ncons."
}

METHODS_MAP_SHORT = {
    "verbal_conf": "VC",
    "ridge_clip": "Ours",
    "split_isotonic_on_ridge": "Ours",
    "split_isotonic_on_ridge_nrt": "Ours",
    "logprob": "TP",
    "ans_logprob": "Ans. TP",
    "oracle_sc": "TT SC"
}

DATASETS_MAP = {
    "gsm8k": "GSM8K",
    "polymath": "Polymath",
    "trivia_qa": "Trivia QA",
    "sciq": "SciQ",
    "webq":"WebQ"
}


def plot_avg_and_worstcase_by_method(
    df: pd.DataFrame,
    metrics=("ECE", "Brier", "AUROC"),
    method_col="Method",
    agg_for_worst=None,          # dict metric -> "max"/"min"; defaults: max for ECE/Brier, min for AUROC/Accuracy
    method_order=None,
    figsize=(13, 3),
    title_prefix="",
    rotate_xticks=20,
    show_values=False,
    plt_save_key=None,
):
    """
    Visualize (1) average and (2) worst-case results across all rows, grouped by Method.

    - Average: mean(metric) within each method.
    - Worst-case:
        * for "lower is better" metrics (ECE, Brier): max
        * for "higher is better" metrics (AUROC, Accuracy): min
      You can override with agg_for_worst={"ECE":"max", "AUROC":"min", ...}

    Produces two figures:
      - "Average" figure with 1 row of subplots (one per metric)
      - "Worst-case" figure with 1 row of subplots (one per metric)
    """

    if agg_for_worst is None:
        agg_for_worst = {}
    # Default directionality
    default_worst = {}
    for m in metrics:
        ml = m.lower()
        if ("ece" in ml) or ("mce" in ml) or ("brier" in ml) or ("nll" in ml) or ("loss" in ml):
            default_worst[m] = "max"   # worst is larger
        else:
            default_worst[m] = "min"   # worst is smaller (AUROC/Accuracy etc.)

    for m in metrics:
        if m not in df.columns:
            raise ValueError(f"Missing metric column '{m}' in df.")
    if method_col not in df.columns:
        raise ValueError(f"Missing method column '{method_col}' in df.")

    # Order methods
    methods = method_order if method_order is not None else sorted(df[method_col].unique().tolist())

    # Average table
    avg = df.groupby(method_col)[list(metrics)].mean().reindex(methods)

    # Worst-case table: per metric choose max/min
    worst = pd.DataFrame(index=methods, columns=list(metrics), dtype=float)
    for m in metrics:
        how = agg_for_worst.get(m, default_worst[m])
        if how == "max":
            worst[m] = df.groupby(method_col)[m].max().reindex(methods)
        elif how == "min":
            worst[m] = df.groupby(method_col)[m].min().reindex(methods)
        else:
            raise ValueError(f"agg_for_worst[{m}] must be 'max' or 'min', got {how!r}")

    def _plot_table(table: pd.DataFrame, suptitle: str):
        n = len(metrics)
        fig, axes = plt.subplots(1, n, figsize=(figsize[0], figsize[1]), constrained_layout=True)
        if n == 1:
            axes = [axes]

        for ax, m in zip(axes, metrics):
            y = table[m].astype(float).values
            x = np.arange(len(methods))

            if len(y) == 5:
                hatch = ["","","","","x"]
                ax.bar(x, y, color=pal, hatch=hatch)
            else:
                ax.bar(x, y, color=pal)

            ax.set_ylim(None, np.max(y)*1.1)

            ax.set_title(m)
            ax.set_xticks(x)
            # ax.set_xticklabels([METHODS_MAP_SQUEEZE[v] for v in methods], rotation=rotate_xticks)
            ax.set_xticklabels([METHODS_MAP_SHORT[v] for v in methods], rotation=rotate_xticks)
            ax.grid(axis="y", alpha=0.3)

            if show_values:
                for xi, yi in zip(x, y):
                    if np.isfinite(yi):
                        ax.text(xi, yi, f"{yi:.3f}", ha="center", va="bottom", fontsize=9)

        fig.suptitle(suptitle, fontsize=13)
        return fig, axes

    fig_avg, _ = _plot_table(avg, f"{title_prefix}Average across all (model, dataset) per method")
    
    if plt_save_key is not None:
        plt_save_path = f"../plots/{plt_save_key}_avg.png"
        plt.savefig(plt_save_path, dpi=600, bbox_inches="tight")

    fig_worst, _ = _plot_table(worst, f"{title_prefix}Worst-case across all (model, dataset) per method")
    
    if plt_save_key is not None:
        plt_save_path = f"../plots/{plt_save_key}_worst.png"
        plt.savefig(plt_save_path, dpi=600, bbox_inches="tight")

    plt.show()


def reliability_diagram_sklearn(
    ax,
    probs,
    correct,
    n_bins=15,
    title="",
    binning="uniform",   # "uniform" (equal-width) or "quantile" (equal-mass)
    show_counts=True,
    limit_axes=False
):
    """
    Reliability diagram using sklearn.calibration.calibration_curve.

    binning:
      - "uniform": equal-width bins on [0,1]
      - "quantile": equal-mass (quantile) bins
    """
    p = np.asarray(probs, dtype=float)
    y = np.asarray(correct, dtype=float)

    # Clean + clip
    m = np.isfinite(p) & np.isfinite(y)
    p = np.clip(p[m], 0.0, 1.0)
    y = y[m].astype(int)

    if len(p) == 0:
        ax.set_title(title + " (no data)")
        ax.set_xlim(0, 1); ax.set_ylim(0, 1)
        return

    prob_true, prob_pred = calibration_curve(y, p, n_bins=n_bins, strategy=binning)

    # Plot curve + y=x
    ax.plot([0, 1], [0, 1], linestyle="--", linewidth=1)
    ax.plot(prob_pred, prob_true, marker="o", linewidth=2)

    if limit_axes:
        ax.set_xlim(np.min(prob_pred)*0.95, None)
        ax.set_ylim(np.min(prob_true)*0.95, None)
    else:
        ax.set_xlim(0, 1)
        ax.set_ylim(0, 1)

    ax.set_xlabel("Predicted probability")
    ax.set_ylabel("Empirical accuracy")
    ax.set_title(f"{title}")
    ax.grid(alpha=0.3)


    # Optional: counts per bin (computed with the same binning rule)
    if show_counts:
        ax2 = ax.twinx()

        if binning == "uniform":
            edges = np.linspace(0.0, 1.0, n_bins + 1)
        elif binning == "quantile":
            # sklearn's quantile strategy uses quantiles of y_prob; do the same
            edges = np.quantile(p, np.linspace(0.0, 1.0, n_bins + 1))
            # Make monotone + clamp endpoints
            edges = np.maximum.accumulate(edges)
            edges[0] = 0.0
            edges[-1] = 1.0
        else:
            raise ValueError("binning must be 'uniform' or 'quantile'")

        # digitize for counts; tolerate duplicate edges by nudging
        edges2 = edges.copy()
        for i in range(1, len(edges2)):
            if edges2[i] <= edges2[i - 1]:
                edges2[i] = np.nextafter(edges2[i - 1], 1.0)
        edges2[0] = 0.0
        edges2[-1] = 1.0

        idx = np.clip(np.digitize(p, edges2, right=False) - 1, 0, n_bins - 1)
        counts = np.bincount(idx, minlength=n_bins)

        centers = 0.5 * (edges2[:-1] + edges2[1:])
        widths = np.clip(edges2[1:] - edges2[:-1], 1e-12, None) * 0.9
        ax2.bar(centers, counts, width=widths, alpha=0.25)
        ax2.set_ylim(0, max(1, int(counts.max() * 1.15)))
        ax2.set_ylabel("Count", rotation=270, labelpad=14)


def plot_reliability(
    con_scores,
    correct,
    logprobs,
    ans_logprobs,
    verbal_conf=None,        # <-- NEW: array in [0,1] aligned with correct, or None
    n_bins=15,
    binning="quantile",
    figsize=None,            # if None, auto based on #panels
    suptitle="Reliability diagrams",
    show_counts=False,
    limit_axes=False,
):
    """
    Reliability diagrams using sklearn calibration_curve.

    Panels:
      - Self-consistency (con_scores in [0,1])
      - exp(avg_logprobs)
      - exp(ans_logprobs)
      - (optional) verbal_conf in [0,1]
    """
    con_prob = np.asarray(con_scores, dtype=float)
    lp_prob  = np.exp(np.asarray(logprobs, dtype=float))
    alp_prob = np.exp(np.asarray(ans_logprobs, dtype=float))

    if verbal_conf is not None:
        vc_prob = np.asarray(verbal_conf, dtype=float)
        panels = [
            ("Logprobs", lp_prob),
            ("Ans. Logprobs", alp_prob),
            ("Verbal Conf.", vc_prob),
            ("Ours", con_prob),
        ]
    else:
        panels = [
            ("Logprobs", lp_prob),
            ("Ans. Logprobs", alp_prob),
            ("Ours", con_prob),
        ]

    n_panels = len(panels)
    if figsize is None:
        figsize = (4.3 * n_panels, 3.0)

    fig, axes = plt.subplots(1, n_panels, figsize=figsize, constrained_layout=True)
    axes = np.atleast_1d(axes).ravel()

    for ax, (title, p) in zip(axes, panels):
        reliability_diagram_sklearn(
            ax,
            probs=p,
            correct=correct,
            n_bins=n_bins,
            title=title,
            binning=binning,
            show_counts=show_counts,
            limit_axes=limit_axes,
        )

    fig.suptitle(suptitle, fontsize=14)
    plt.show()


def plot_method_comparisons_by_dataset(
    full_df: pd.DataFrame,
    metrics,
    datasets,
    *,
    models=None,              # None or list/str to filter Model
    method_order=None,        # optional explicit ordering of Method bars
    agg="mean",               # "mean" | "median"
    err="sem",                # None | "std" | "sem"  (computed over replicates)
    replicate_cols=("Model", "seed"),  # what defines independent replicates
    figsize_per_cell=(3.6, 2.6),
    sharey="row",             # True | False | "row" | "col"
    ylim=None,                # None or (lo, hi)
):
    """
    Grid of bar plots:
      - columns = datasets
      - rows    = metrics
      - bars    = methods (aggregated over replicate_cols, optionally filtered by models)

    Assumes columns like: ["Model","Dataset","Method", <metrics...>, "seed"].
    """

    metrics = list(metrics)
    datasets = list(datasets)

    df = full_df.copy()

    # Filter datasets/models
    df = df[df["Dataset"].isin(datasets)]
    if models is not None:
        if isinstance(models, str):
            models = [models]
        df = df[df["Model"].isin(models)]

    # Decide method order
    if method_order is None:
        method_order = sorted(df["Method"].dropna().unique().tolist())
    else:
        method_order = list(method_order)

    # Aggregation function
    if agg == "mean":
        agg_fn = "mean"
    elif agg == "median":
        agg_fn = "median"
    else:
        raise ValueError(f"Unknown agg={agg}. Use 'mean' or 'median'.")

    # Build figure
    nrows, ncols = len(metrics), len(datasets)
    figsize = (figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)

    # Make axes always 2D for consistent indexing
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # Helper: compute error from replicate-level values
    def _errbar(vals: pd.Series):
        if err is None:
            return np.nan
        vals = vals.dropna().to_numpy()
        if vals.size <= 1:
            return 0.0
        if err == "std":
            return float(np.std(vals, ddof=1))
        if err == "sem":
            return float(np.std(vals, ddof=1) / np.sqrt(vals.size))
        raise ValueError(f"Unknown err={err}. Use None, 'std', or 'sem'.")

    for r, metric in enumerate(metrics):
        for c, dataset in enumerate(datasets):
            ax = axes[r, c]

            sub = df[df["Dataset"] == dataset][["Method", metric, *replicate_cols]].dropna(subset=[metric])

            # replicate-level aggregation first (so error bars reflect replicate variation)
            # e.g., average within (Model, seed, Method) then aggregate across replicates
            rep = (
                sub.groupby(list(replicate_cols) + ["Method"], as_index=False)[metric]
                   .agg(agg_fn)
            )

            # method-level summary
            mean_tbl = rep.groupby("Method")[metric].agg(agg_fn).reindex(method_order)
            err_tbl = rep.groupby("Method")[metric].apply(_errbar).reindex(method_order)

            x = np.arange(len(method_order))
            y = mean_tbl.to_numpy(dtype=float)
            yerr = None if err is None else err_tbl.to_numpy(dtype=float)

            ax.bar(x, y, yerr=yerr if err is not None else None, capsize=3, color=pal)

            ax.set_xticks(x)
            ax.set_xticklabels(method_order, rotation=45, ha="right")

            if r == 0:
                ax.set_title(dataset)
            if c == 0:
                ax.set_ylabel(metric)

            if ylim is not None:
                ax.set_ylim(*ylim)

            ax.grid(axis="y", alpha=0.3)

    fig.tight_layout()
    plt.show()


def plot_method_comparisons(
    df: pd.DataFrame,
    metrics,
    *,
    facet="Dataset",              # "Dataset" or "Model" (what goes on columns)
    dataset_order=None,           # optional: order + subset datasets
    model_order=None,             # optional: order + subset models
    method_order=None,            # optional: order + subset methods
    agg="mean",                   # "mean" | "median"
    err="sem",                    # None | "std" | "sem"
    replicate_cols=("seed",),     # what defines independent replicates (often just "seed")
    figsize_per_cell=(3.6, 2.6),
    sharey="row",                 # True | False | "row" | "col"
    ylim=None,                    # None or (lo, hi)
    plt_title=None
):
    """
    Grid of bar plots:
      - columns = unique values of `facet` ("Dataset" or "Model")
      - rows    = metrics
      - bars    = Method (aggregated over replicate_cols, optionally filtered by model/dataset orders)

    Defaults:
      - includes ALL models/datasets present in df
      - dataset_order/model_order/method_order, if provided, both ORDER and FILTER.
    """

    metrics = list(metrics)
    if facet not in ("Dataset", "Model"):
        raise ValueError("facet must be 'Dataset' or 'Model'")

    data = df.copy()

    # Apply optional filters + ordering for Dataset / Model / Method
    if dataset_order is None:
        dataset_vals = sorted(data["Dataset"].dropna().unique().tolist())
    else:
        dataset_vals = list(dataset_order)
        data = data[data["Dataset"].isin(dataset_vals)]

    if model_order is None:
        model_vals = sorted(data["Model"].dropna().unique().tolist())
    else:
        model_vals = list(model_order)
        data = data[data["Model"].isin(model_vals)]

    if method_order is None:
        method_vals = sorted(data["Method"].dropna().unique().tolist())
    else:
        method_vals = list(method_order)
        data = data[data["Method"].isin(method_vals)]

    # Choose which values become columns
    if facet == "Dataset":
        col_vals = dataset_vals
    else:  # facet == "Model"
        col_vals = model_vals

    # Aggregation function
    if agg == "mean":
        agg_fn = "mean"
    elif agg == "median":
        agg_fn = "median"
    else:
        raise ValueError("agg must be 'mean' or 'median'")

    # Error-bar helper over replicate-level values
    def _errbar(vals: pd.Series):
        if err is None:
            return np.nan
        vals = vals.dropna().to_numpy()
        if vals.size <= 1:
            return 0.0
        if err == "std":
            return float(np.std(vals, ddof=1))
        if err == "sem":
            return float(np.std(vals, ddof=1) / np.sqrt(vals.size))
        raise ValueError("err must be None, 'std', or 'sem'")

    # Create figure
    nrows, ncols = len(metrics), len(col_vals)
    figsize = (figsize_per_cell[0] * ncols, figsize_per_cell[1] * nrows)
    fig, axes = plt.subplots(nrows, ncols, figsize=figsize, sharey=sharey)

    # force 2D axes
    if nrows == 1 and ncols == 1:
        axes = np.array([[axes]])
    elif nrows == 1:
        axes = axes.reshape(1, -1)
    elif ncols == 1:
        axes = axes.reshape(-1, 1)

    # Main loop
    for r, metric in enumerate(metrics):
        for c, col_val in enumerate(col_vals):
            ax = axes[r, c]

            sub = data[data[facet] == col_val][["Model", "Dataset", "Method", metric, *replicate_cols]].dropna(subset=[metric])

            # replicate-level aggregation first
            # (e.g., if df has multiple rows per seed/method, we aggregate within each replicate)
            rep = (
                sub.groupby(list(replicate_cols) + ["Method"], as_index=False)[metric]
                   .agg(agg_fn)
            )

            # method-level summary across replicates
            y_tbl = rep.groupby("Method")[metric].agg(agg_fn).reindex(method_vals)
            e_tbl = rep.groupby("Method")[metric].apply(_errbar).reindex(method_vals)

            x = np.arange(len(method_vals))
            y = y_tbl.to_numpy(dtype=float)
            yerr = None if err is None else e_tbl.to_numpy(dtype=float)

            ax.bar(x, y, yerr=yerr, capsize=3 if err is not None else 0, color=pal)

            ax.set_xticks(x)
            ax.set_xticklabels([METHODS_MAP_SQUEEZE[v] for v in method_vals], rotation=20)

            if r == 0:
                if facet == "Model":
                    c_title = col_val.split("/")[-1]
                else:
                    if col_val in DATASETS_MAP:
                        c_title = DATASETS_MAP[col_val]
                    else:
                        c_title = col_val
                ax.set_title(c_title)
            if c == 0:
                ax.set_ylabel(metric)

            if ylim is not None:
                ax.set_ylim(*ylim)

            ax.grid(axis="y", alpha=0.3)

    if plt_title is not None:
        fig.suptitle(plt_title)

    fig.tight_layout()
    plt.show()