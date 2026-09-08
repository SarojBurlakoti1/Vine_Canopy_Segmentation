"""
Plant-clustered pairwise model comparison using per-image macro-F1 on the 169-image test set.

Clustering variable: `group_id` from split_assignment_final.csv (the same field used
throughout this project to identify shared plant/treatment identity -- tier A =
individually block-resolved plants, tier B = shared treatment x species groups, tiers
C/D = independent "INDEP|filename" singleton photos).

IMPORTANT finding, checked explicitly before running anything: on the actual 169-image
TEST split, every group_id is unique (169 distinct clusters for 169 images) -- tier B
(the tier with real multi-image clusters, 20-40 images per group) has ZERO images in the
test split; only tiers A/C/D appear there, and tier A is already one photo per plant. So
a "plant-clustered" bootstrap on this specific test set is mathematically identical to a
plain per-image bootstrap (every cluster has size 1). The code below implements genuine
cluster resampling (so it would matter if run on train/val, or on a future test set with
repeated plants), but on this test set it is a no-op relative to i.i.d. image resampling.
This is reported explicitly in the output rather than silently treating cluster-bootstrap
results as if they used information that, here, they didn't have available to use.

Method: for each seed and each of the 21 model pairs, compute the observed mean paired
difference in per-image F1, then bootstrap by resampling clusters (whole clusters, all
member images together) with replacement B=10000 times, building the bootstrap
distribution of the mean paired difference. Two-sided bootstrap p-value via the
percentile method (2*min(P(diff<=0), P(diff>=0)), capped at 1). Effect size: raw mean
F1 difference, plus paired Cohen's d (mean diff / SD of per-image differences). Holm
correction applied across the 21 pairwise p-values within each seed.
"""
import json
import itertools
import numpy as np
import pandas as pd

RUN = "/root/vine-project/Run"
SEEDS = [42, 43, 44]
MODELS = ["UNet", "DeepLabV3", "SegFormer", "MaskRCNN", "SoloV2", "YOLO11seg", "PGSS_YOLO11"]
DISPLAY = {"UNet": "UNet", "DeepLabV3": "DeepLabV3", "SegFormer": "SegFormer", "MaskRCNN": "Mask R-CNN",
           "SoloV2": "SOLOv2", "YOLO11seg": "YOLO11-seg", "PGSS_YOLO11": "YOLO11-PGSS"}
B = 10000
RNG = np.random.default_rng(20260904)


def log(*a):
    print(*a, flush=True)


# ---------------------------------------------------------------------------
# Load per-image F1 and cluster (group_id) assignment
# ---------------------------------------------------------------------------
per_image_f1 = json.load(open(f"{RUN}/per_image_f1.json"))
split_df = pd.read_csv(f"{RUN}/split_assignment_final.csv")
fname_to_group = dict(zip(split_df["roboflow_filename"], split_df["group_id"]))

per_image_csv = pd.read_csv(f"{RUN}/per_image_f1.csv")
fname_by_iid = per_image_csv.drop_duplicates("image_id").set_index("image_id")["file_name"].to_dict()

# sanity: cluster structure on the test set
test_iids = sorted(fname_by_iid.keys())
test_groups = [fname_to_group[fname_by_iid[iid]] for iid in test_iids]
n_unique_groups = len(set(test_groups))
log(f"Test set: {len(test_iids)} images, {n_unique_groups} distinct group_id clusters "
    f"({'all singleton -- clustering is a no-op here' if n_unique_groups == len(test_iids) else 'some multi-image clusters present'})")

cluster_of_iid = {iid: fname_to_group[fname_by_iid[iid]] for iid in test_iids}


def build_wide(seed):
    """Return DataFrame: index=image_id, columns=model, values=f1; plus cluster series."""
    cols = {}
    for m in MODELS:
        d = per_image_f1[m][str(seed)]
        cols[m] = {int(iid): f1 for iid, f1 in d.items()}
    df = pd.DataFrame(cols)
    df = df.loc[test_iids]  # fixed, consistent image order
    df["cluster"] = [cluster_of_iid[iid] for iid in df.index]
    return df


def cluster_bootstrap_pair(df, model_a, model_b, B=B, rng=RNG):
    diffs = (df[model_a] - df[model_b]).values
    clusters = df["cluster"].values
    unique_clusters = np.unique(clusters)
    # map cluster -> indices (list, since a cluster may have >1 image in general)
    cluster_to_idx = {c: np.where(clusters == c)[0] for c in unique_clusters}
    n_clusters = len(unique_clusters)

    observed = float(diffs.mean())
    sd_diff = float(diffs.std(ddof=1)) if len(diffs) > 1 else np.nan
    cohens_d = observed / sd_diff if sd_diff and sd_diff > 0 else np.nan

    boot_means = np.empty(B)
    for b in range(B):
        sampled_clusters = rng.choice(unique_clusters, size=n_clusters, replace=True)
        idx = np.concatenate([cluster_to_idx[c] for c in sampled_clusters])
        boot_means[b] = diffs[idx].mean()

    ci_lo, ci_hi = np.percentile(boot_means, [2.5, 97.5])
    p_le = np.mean(boot_means <= 0)
    p_ge = np.mean(boot_means >= 0)
    p_raw = min(1.0, 2 * min(p_le, p_ge))

    return {
        "mean_diff_f1": observed, "cohens_d_paired": cohens_d,
        "boot_ci_lo": float(ci_lo), "boot_ci_hi": float(ci_hi), "p_raw": float(p_raw),
    }


def holm_correct(pvals):
    """Return Holm-adjusted p-values, same order as input."""
    idx_sorted = np.argsort(pvals)
    m = len(pvals)
    adj = np.empty(m)
    running_max = 0.0
    for rank, idx in enumerate(idx_sorted):
        val = (m - rank) * pvals[idx]
        running_max = max(running_max, val)
        adj[idx] = min(1.0, running_max)
    return adj


# ---------------------------------------------------------------------------
pairs = list(itertools.combinations(MODELS, 2))
log(f"{len(pairs)} pairwise comparisons per seed")

all_rows = []
for seed in SEEDS:
    log(f"=== seed {seed} ===")
    df = build_wide(seed)
    raw_pvals = []
    pair_results = []
    for a, b in pairs:
        res = cluster_bootstrap_pair(df, a, b)
        pair_results.append((a, b, res))
        raw_pvals.append(res["p_raw"])
    adj_pvals = holm_correct(np.array(raw_pvals))
    for (a, b, res), p_holm in zip(pair_results, adj_pvals):
        row = {
            "seed": seed, "model_a": DISPLAY[a], "model_b": DISPLAY[b],
            "mean_diff_f1_a_minus_b": res["mean_diff_f1"], "cohens_d_paired": res["cohens_d_paired"],
            "boot_ci95_lo": res["boot_ci_lo"], "boot_ci95_hi": res["boot_ci_hi"],
            "p_raw": res["p_raw"], "p_holm": float(p_holm),
            "significant_holm_0.05": bool(p_holm < 0.05),
            "direction": ("a_higher" if res["mean_diff_f1"] > 0 else "b_higher" if res["mean_diff_f1"] < 0 else "tie"),
        }
        all_rows.append(row)
    log(f"  seed {seed}: {sum(r['significant_holm_0.05'] for r in all_rows if r['seed']==seed)}/{len(pairs)} pairs significant (Holm 0.05)")

results_df = pd.DataFrame(all_rows)
results_df.to_csv(f"{RUN}/pairwise_stats_per_seed.csv", index=False)
log(f"Wrote pairwise_stats_per_seed.csv ({len(results_df)} rows)")

# ---------------------------------------------------------------------------
# Cross-seed consistency check
# ---------------------------------------------------------------------------
consistency_rows = []
for a, b in pairs:
    sub = results_df[(results_df["model_a"] == DISPLAY[a]) & (results_df["model_b"] == DISPLAY[b])]
    directions = set(sub["direction"])
    sig_flags = dict(zip(sub["seed"], sub["significant_holm_0.05"]))
    p_holms = dict(zip(sub["seed"], sub["p_holm"]))
    diffs = dict(zip(sub["seed"], sub["mean_diff_f1_a_minus_b"]))
    consistency_rows.append({
        "model_a": DISPLAY[a], "model_b": DISPLAY[b],
        "direction_consistent_across_seeds": len(directions) == 1,
        "significant_in_all_3_seeds": all(sig_flags.values()),
        "significant_in_any_seed": any(sig_flags.values()),
        "diff_seed42": diffs.get(42), "diff_seed43": diffs.get(43), "diff_seed44": diffs.get(44),
        "p_holm_seed42": p_holms.get(42), "p_holm_seed43": p_holms.get(43), "p_holm_seed44": p_holms.get(44),
    })
consistency_df = pd.DataFrame(consistency_rows)
consistency_df.to_csv(f"{RUN}/pairwise_stats_consistency.csv", index=False)
log(f"Wrote pairwise_stats_consistency.csv ({len(consistency_df)} rows)")

n_dir_consistent = consistency_df["direction_consistent_across_seeds"].sum()
n_sig_all3 = consistency_df["significant_in_all_3_seeds"].sum()
log(f"\nSummary: {n_dir_consistent}/{len(pairs)} pairs have the same direction in all 3 seeds; "
    f"{n_sig_all3}/{len(pairs)} pairs are Holm-significant in ALL 3 seeds.")

# also dump a json with full detail + the methodology note
out = {
    "methodology": {
        "clustering_variable": "group_id (split_assignment_final.csv)",
        "cluster_structure_on_test_set": f"{len(test_iids)} images, {n_unique_groups} distinct clusters "
                                          "(all singleton -- see script docstring)",
        "bootstrap_reps": B,
        "p_value_method": "two-sided percentile bootstrap: 2*min(P(diff<=0),P(diff>=0)), capped at 1",
        "multiple_comparison_correction": "Holm, within each seed across the 21 pairwise comparisons",
        "effect_sizes_reported": ["mean_diff_f1 (raw, interpretable directly)", "cohens_d_paired"],
    },
    "per_seed_results": all_rows,
    "cross_seed_consistency": consistency_rows,
}
with open(f"{RUN}/pairwise_stats_full.json", "w") as f:
    json.dump(out, f, indent=2)
log("Wrote pairwise_stats_full.json")
