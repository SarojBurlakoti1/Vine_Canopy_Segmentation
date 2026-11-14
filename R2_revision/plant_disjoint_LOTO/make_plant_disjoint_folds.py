"""
make_plant_disjoint_folds.py
============================
Builds a VERIFIABLY plant-disjoint evaluation design from split_assignment_final.csv.

Design (primary, verifiable with existing metadata):
  Leave-one-treatment-out (LOTO), 3 folds, using only images whose treatment is known
  (tier A = block-resolved plants; tier B = treatment x species groups). Plants receiving
  different treatments are physically different individuals, so the held-out treatment's
  plants never appear in training or validation. Tier C (2023, dated) and tier D (undated)
  images are EXCLUDED from training and testing in this design because their treatment/plant
  identity is unknown and they are repeat photographs of the same trial plants.

Validation (checkpoint selection) inside each fold:
  Tier-A images from the two training treatments are split by BLOCK (default blocks 6-8 to
  validation). Tier-B images of the training treatments go to training. Validation therefore
  shares plants with training (affects epoch selection only); the TEST fold is strictly
  plant-disjoint from both training and validation.

Optional (stronger, if identity is recovered for tier B):
  Supply --tierB_block_map CSV (columns: roboflow_filename, block) to run leave-one-block-out
  or block-held-out designs instead.

Usage:
  python -I make_plant_disjoint_folds.py split_assignment_final.csv OUT_DIR [--val_blocks 6 7 8]

Outputs:
  OUT_DIR/loto_fold_assignment.csv  (roboflow_filename, species, tier, treatment, block, fold_k_split ...)
  OUT_DIR/loto_fold_summary.json    (counts per fold/split/species, disjointness assertions)
"""
import argparse, json, sys
import pandas as pd

TIER_A, TIER_B = "A_block_resolved", "B_treatment_species"


def parse(df):
    df = df.copy()
    df["treatment"] = pd.NA
    df["block"] = pd.NA
    a = df.tier == TIER_A
    parts = df.loc[a, "group_id"].str.split("|", expand=True)  # BLOCK|block|trt|species
    df.loc[a, "block"] = parts[1].astype(int)
    df.loc[a, "treatment"] = parts[2]
    b = df.tier == TIER_B
    df.loc[b, "treatment"] = df.loc[b, "group_id"].str.split("|").str[1]  # TRTSP|trt|species
    return df


def main():
    ap = argparse.ArgumentParser()
    ap.add_argument("manifest"); ap.add_argument("out_dir")
    ap.add_argument("--val_blocks", nargs="+", type=int, default=[6, 7, 8])
    a = ap.parse_args()
    df = parse(pd.read_csv(a.manifest))
    ab = df[df.tier.isin([TIER_A, TIER_B])].copy()
    treatments = sorted(ab.treatment.dropna().unique())
    summary = {"design": "leave-one-treatment-out on tiers A+B", "treatments": treatments,
               "excluded_tiers_C_D": int((~df.tier.isin([TIER_A, TIER_B])).sum()), "folds": {}}
    for t in treatments:
        col = f"fold_test_trt{t}"
        s = pd.Series("train", index=ab.index)
        s[ab.treatment == t] = "test"
        s[(ab.treatment != t) & (ab.tier == TIER_A) & (ab.block.isin(a.val_blocks))] = "val"
        ab[col] = s
        # --- disjointness assertions (plant identity = block|treatment|species for tier A;
        #     treatment|species group for tier B; treatments are disjoint sets of plants) ---
        test_trts = set(ab.loc[s == "test", "treatment"]); other_trts = set(ab.loc[s != "test", "treatment"])
        assert test_trts.isdisjoint(other_trts), "treatment leakage between test and train/val"
        test_groups = set(ab.loc[s == "test", "group_id"]); other_groups = set(ab.loc[s != "test", "group_id"])
        assert test_groups.isdisjoint(other_groups), "group_id leakage between test and train/val"
        summary["folds"][f"test_treatment_{t}"] = {
            "n_by_split": s.value_counts().to_dict(),
            "test_by_species": ab.loc[s == "test", "species"].value_counts().sort_index().to_dict(),
            "train_by_species": ab.loc[s == "train", "species"].value_counts().sort_index().to_dict(),
            "val_by_species": ab.loc[s == "val", "species"].value_counts().sort_index().to_dict(),
            "test_tierA_plants": int(((s == "test") & (ab.tier == TIER_A)).sum()),
            "assert_test_disjoint_from_train_and_val": True,
        }
    keep = ["roboflow_filename", "species", "tier", "group_id", "treatment", "block"] + [c for c in ab.columns if c.startswith("fold_")]
    ab[keep].to_csv(f"{a.out_dir}/loto_fold_assignment.csv", index=False)
    json.dump(summary, open(f"{a.out_dir}/loto_fold_summary.json", "w"), indent=1, default=str)
    print(json.dumps(summary, indent=1, default=str))


if __name__ == "__main__":
    main()
