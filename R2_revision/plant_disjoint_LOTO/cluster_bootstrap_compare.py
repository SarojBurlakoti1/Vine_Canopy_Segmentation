"""
cluster_bootstrap_compare.py
============================
Pairwise model comparison on per-image scores with cluster bootstrap, Holm correction and an
equivalence check (for statements such as "not meaningfully different").

INPUT per_image.csv with columns:
    image_id, cluster_id, fold (or seed), model, score
  cluster_id = treatment x species group for ALL tier A and tier B images (21 groups over the
  three LOTO folds). A tier-A plant is also one of the eight plants of its tier-B group, so
  plant-level clusters for tier A and group-level clusters for tier B would NOT be independent.
  Use plant-level clusters only if block identity has been recovered for every tier-B image.
  score = per-image macro F1 or per-image mIoU (classes absent from both GT and prediction excluded).

Method:
  * For each model pair, per-image paired difference d_i = score_A - score_B (same image/fold).
  * Resample CLUSTERS with replacement (B = 10,000), pooling all folds; statistic = mean d.
  * 95% percentile CI; two-sided p = 2 * min(P(d* <= 0), P(d* >= 0)) (capped at 1); Holm across pairs.
  * Equivalence (TOST-equivalent): pair declared equivalent at margin delta if the 90% bootstrap CI
    lies entirely inside (-delta, +delta). delta must be PRE-SPECIFIED (e.g. 0.02 = 2 points).
Run `python -I cluster_bootstrap_compare.py --selftest` (synthetic data; not a result).
"""
import argparse, itertools
import numpy as np
import pandas as pd


def holm(p):
    p = np.asarray(p); o = np.argsort(p); m = len(p); adj = np.empty(m); run = 0
    for r, i in enumerate(o):
        run = max(run, (m - r) * p[i]); adj[i] = min(1.0, run)
    return adj


def compare(df, B=10000, delta=0.02, seed=20261106):
    rng = np.random.default_rng(seed)
    wide = df.pivot_table(index=["fold", "image_id", "cluster_id"], columns="model", values="score").reset_index()
    models = [c for c in wide.columns if c not in ("fold", "image_id", "cluster_id")]
    clusters = wide.cluster_id.unique(); idx = {c: np.where(wide.cluster_id.values == c)[0] for c in clusters}
    draws = [np.concatenate([idx[c] for c in rng.choice(clusters, len(clusters), replace=True)]) for _ in range(B)]
    rows = []
    for a, b in itertools.combinations(models, 2):
        d = (wide[a] - wide[b]).values
        boot = np.array([d[k].mean() for k in draws])
        rows.append(dict(model_a=a, model_b=b, n_images=len(d), n_clusters=len(clusters), mean_diff=d.mean(),
                         ci95_lo=np.percentile(boot, 2.5), ci95_hi=np.percentile(boot, 97.5),
                         ci90_lo=np.percentile(boot, 5), ci90_hi=np.percentile(boot, 95),
                         p_raw=min(1.0, 2 * min((boot <= 0).mean(), (boot >= 0).mean()))))
    out = pd.DataFrame(rows); out["p_holm"] = holm(out.p_raw.values)
    out["equivalent_within_delta"] = (out.ci90_lo > -delta) & (out.ci90_hi < delta); out["delta"] = delta
    return out


def _selftest():
    rng = np.random.default_rng(3); rows = []
    for f in range(3):
        for c in range(21):
            eff = rng.normal(0, 0.03)
            for i in range(9):
                base = 0.8 + eff + rng.normal(0, 0.02)
                for m, shift in [("A", 0.0), ("B", 0.0), ("C", -0.08)]:
                    rows.append(dict(fold=f, image_id=f"{f}_{c}_{i}", cluster_id=f"g{c}", model=m, score=base + shift + rng.normal(0, 0.01)))
    out = compare(pd.DataFrame(rows), B=2000)
    ab = out[(out.model_a == "A") & (out.model_b == "B")].iloc[0]; ac = out[(out.model_a == "A") & (out.model_b == "C")].iloc[0]
    assert ab.p_holm > 0.05 and ab.equivalent_within_delta and ac.p_holm < 0.05 and not ac.equivalent_within_delta
    print("cluster_bootstrap_compare self-test passed (synthetic data only; not a result).")


if __name__ == "__main__":
    ap = argparse.ArgumentParser(); ap.add_argument("--per_image_csv"); ap.add_argument("--out_csv")
    ap.add_argument("--delta", type=float, default=0.02); ap.add_argument("--selftest", action="store_true")
    a = ap.parse_args()
    if a.selftest:
        _selftest()
    else:
        compare(pd.read_csv(a.per_image_csv), delta=a.delta).to_csv(a.out_csv, index=False)
