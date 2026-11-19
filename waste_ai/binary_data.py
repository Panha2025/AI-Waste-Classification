"""Prepare a separate two-class experiment from the audited public manifest.

Targets are inferred from source folders and are explicitly not human-verified
recycling decisions. Original images and the six-class manifest are untouched.
"""
from __future__ import annotations

from collections import Counter, defaultdict
import json
import random
from pathlib import Path

import pandas as pd

from waste_ai.config import PROCESSED, SEED

BINARY_LABELS = ("non_recyclable", "recyclable")
SOURCE_TO_BINARY = {
    "cardboard": "recyclable", "paper": "recyclable", "glass": "recyclable",
    "metal": "recyclable", "plastic": "recyclable", "trash": "non_recyclable",
}
INPUT = PROCESSED / "split_manifest.csv"
OUTPUT = PROCESSED / "binary_split_manifest.csv"

# Flagged during the representative visual spot-check; keep source data intact.
KNOWN_AMBIGUOUS_SUFFIXES = {"/trash100.jpg"}


def _group_stratified_splits(frame, seed=SEED, fractions=(0.70, 0.15, 0.15)):
    rng = random.Random(seed)
    assignment = {}
    split_names = ("train", "val", "test")
    for label in BINARY_LABELS:
        subset = frame[frame.binary_label == label]
        groups = subset.groupby("capture_group", sort=True).size().to_dict()
        if len(groups) < 3:
            raise ValueError(f"Need at least 3 independent groups for {label}; found {len(groups)}")
        items = list(groups.items())
        rng.shuffle(items)
        items.sort(key=lambda item: item[1], reverse=True)
        targets = [len(subset) * fraction for fraction in fractions]
        counts = [0, 0, 0]
        for index, (group, size) in enumerate(items):
            empty = [i for i, count in enumerate(counts) if count == 0]
            remaining = len(items) - index
            if remaining == len(empty):
                chosen = empty[0]
            else:
                # Greedily minimize normalized squared deviation from target size.
                scores = []
                for i in range(3):
                    after = counts.copy()
                    after[i] += size
                    scores.append(sum(((after[j] - targets[j]) / max(targets[j], 1)) ** 2
                                      for j in range(3)))
                chosen = min(range(3), key=lambda i: scores[i])
            assignment[group] = split_names[chosen]
            counts[chosen] += size
    return assignment


def prepare_binary_data(seed=SEED):
    if not INPUT.is_file():
        raise FileNotFoundError("Run `python -m waste_ai.data` first")
    source = pd.read_csv(INPUT).fillna("")
    source["binary_label"] = source.label.map(SOURCE_TO_BINARY)
    source = source[source.binary_label.notna()].copy()
    source["label_origin"] = "source_folder_mapped_provisional_unverified"
    normalized_paths = source.path.astype(str).str.replace("\\\\", "/", regex=True)
    excluded_mask = (source.source.astype(str).str.endswith("trashnet") &
                     normalized_paths.str.endswith(tuple(KNOWN_AMBIGUOUS_SUFFIXES)))
    exclusions = source.loc[excluded_mask].copy()
    if not exclusions.empty:
        exclusions["exclusion_reason"] = "representative_spot_check_suspected_source_label_ambiguity"
    eligible = source.loc[~excluded_mask].copy()
    eligible["split"] = eligible["capture_group"].map(_group_stratified_splits(eligible, seed))
    if eligible["split"].isna().any():
        raise ValueError("Some images were not assigned to a split")
    for split in ("train", "val", "test"):
        labels = set(eligible.loc[eligible.split == split, "binary_label"])
        if labels != set(BINARY_LABELS):
            raise ValueError(f"Both binary classes must be represented in {split}; found {labels}")
    for group, rows in eligible.groupby("capture_group"):
        if rows.split.nunique() != 1:
            raise ValueError(f"Group leakage detected for {group}")
    if eligible.sha256.duplicated().any():
        raise ValueError("Exact duplicate hashes remain in the binary manifest")
    eligible.to_csv(OUTPUT, index=False)
    exclusions.to_csv(PROCESSED / "binary_label_exclusions.csv", index=False)
    split_counts = (eligible.groupby(["split", "binary_label", "source", "label"],
                                     dropna=False).size().rename("images").reset_index())
    split_counts.to_csv(PROCESSED / "binary_split_counts.csv", index=False)
    class_counts = {label: int(count) for label, count in
                    eligible.binary_label.value_counts().reindex(BINARY_LABELS, fill_value=0).items()}
    split_totals = {str(k): int(v) for k, v in eligible.split.value_counts().items()}
    summary = {
        "seed": seed, "n_total": int(len(eligible)), "n_excluded_known_ambiguous": int(len(exclusions)),
        "class_counts": class_counts, "split_counts": split_totals,
        "split_by_class": {str(k): {str(kk): int(vv) for kk, vv in val.items()}
                           for k, val in eligible.groupby("split").binary_label.value_counts().unstack(fill_value=0).iterrows()},
        "human_verified_labels": 0,
        "label_provenance": "inferred from original source category folders; not human-verified",
        "leakage_prevention": "capture groups kept wholly within a split; unresolved near-duplicate candidates excluded upstream",
        "local_test_images": 0,
        "mapping": SOURCE_TO_BINARY,
        "warning": "Metrics measure agreement with provisional mapped source labels, not verified local recyclability decisions.",
    }
    (PROCESSED / "binary_summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    _save_distribution(eligible)
    print(json.dumps(summary, indent=2))
    return eligible, summary


def _save_distribution(frame):
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    counts = frame.groupby(["split", "binary_label"]).size().unstack(fill_value=0).reindex(
        index=("train", "val", "test"), columns=BINARY_LABELS, fill_value=0)
    ax = counts.plot(kind="bar", figsize=(8, 4), color=("#b65f42", "#28745a"))
    ax.set_ylabel("Images")
    ax.set_title("Binary split distribution (provisional source-folder labels)")
    ax.set_xlabel("Split")
    ax.legend(title="Inferred class")
    plt.tight_layout()
    plt.savefig(PROCESSED / "binary_class_distribution.png", dpi=150)
    plt.close()


def main():
    prepare_binary_data()


if __name__ == "__main__":
    main()
