"""Create a reproducible, source/category-stratified human review pilot queue."""
from __future__ import annotations

import argparse
import json
import math
from pathlib import Path

import pandas as pd

from waste_ai.config import PROCESSED, SEED

CANDIDATES = PROCESSED / "candidate_manifest.csv"
NEAR_REVIEW = PROCESSED / "near_duplicate_review.csv"
QUEUE = PROCESSED / "pilot_review_queue.csv"
SUMMARY = PROCESSED / "pilot_review_queue_summary.csv"
EXCLUDED = PROCESSED / "pilot_queue_duplicate_exclusions.csv"


def _stratified_sample(frame: pd.DataFrame, size: int, strata: list[str], seed: int):
    counts = frame.groupby(strata, dropna=False, sort=True).size().rename("available").reset_index()
    if size < len(counts):
        raise ValueError(f"Sample size {size} is smaller than {len(counts)} strata")
    if size > len(frame):
        raise ValueError(f"Requested {size} images but only {len(frame)} are eligible")
    counts["ideal"] = counts.available * (size / len(frame))
    counts["quota"] = counts.ideal.apply(math.floor).astype(int)
    remaining = size - int(counts.quota.sum())
    order = counts.assign(frac=counts.ideal - counts.quota).sort_values(
        ["frac", *strata], ascending=[False, *([True] * len(strata))], kind="stable")
    for idx in order.index[:remaining]:
        counts.loc[idx, "quota"] += 1
    # Ensure every nonempty source/category stratum is represented, taking one slot
    # from the largest over-allocated stratum where possible.
    for idx in counts.index[counts.quota == 0]:
        donors = counts[counts.quota > 1].copy()
        if donors.empty:
            raise ValueError("Cannot represent every source/category stratum")
        donors["excess"] = donors.quota - donors.ideal
        donor = donors.sort_values(["excess", *strata], ascending=[False, *([True] * len(strata))], kind="stable").index[0]
        counts.loc[donor, "quota"] -= 1
        counts.loc[idx, "quota"] = 1

    selected = []
    for _, allocation in counts.iterrows():
        mask = pd.Series(True, index=frame.index)
        for column in strata:
            mask &= frame[column].astype(str).eq(str(allocation[column]))
        group = frame[mask]
        chosen = group.sample(n=int(allocation.quota), random_state=seed)
        selected.append(chosen.assign(stratum_target=int(allocation.quota)))
    return pd.concat(selected).sort_values([*strata, "sha256"], kind="stable")


def build_pilot_queue(target_per_class=300, seed=SEED):
    if not CANDIDATES.exists():
        raise FileNotFoundError(f"Candidate manifest not found: {CANDIDATES}")
    candidates = pd.read_csv(CANDIDATES).fillna("")
    if candidates.sha256.duplicated().any():
        raise ValueError("Candidate manifest contains exact duplicate SHA-256 hashes")
    candidates["proposed_binary_class"] = candidates.proposed_binary_class.astype(str)
    if NEAR_REVIEW.exists():
        near = pd.read_csv(NEAR_REVIEW).fillna("")
        flagged_paths = set(near.path.astype(str)) | set(near.candidate_duplicate_of.astype(str))
    else:
        flagged_paths = set()
    flagged = candidates.original_path.astype(str).isin(flagged_paths)
    if "exact_label_conflict" in candidates.columns:
        flagged |= candidates.exact_label_conflict.astype(str).str.lower().isin({"true", "1", "yes"})
    exclusions = candidates.loc[flagged].copy()
    exclusions["exclusion_reason"] = "unresolved_suspected_near_duplicate"
    if "exact_label_conflict" in exclusions.columns:
        exclusions.loc[exclusions.exact_label_conflict.astype(str).str.lower().isin({"true", "1", "yes"}),
                       "exclusion_reason"] = "unresolved_exact_label_conflict"
    eligible = candidates.loc[~flagged].copy()

    class_frames = []
    for label in ("recyclable", "non_recyclable"):
        subset = eligible[eligible.proposed_binary_class == label]
        if len(subset) < target_per_class:
            raise ValueError(f"Only {len(subset)} eligible {label} candidates; need {target_per_class}")
        strata = ["source", "original_category"] if label == "recyclable" else ["source"]
        sampled = _stratified_sample(subset, target_per_class, strata, seed)
        sampled["sampling_strata"] = ";".join(strata)
        class_frames.append(sampled)
    queue = pd.concat(class_frames).sample(frac=1, random_state=seed).reset_index(drop=True)
    queue.insert(0, "queue_order", range(1, len(queue) + 1))
    queue["sampling_seed"] = seed
    queue["queue_status"] = "provisional_unreviewed"
    queue["selection_method"] = "seeded_stratified_random; unresolved near-duplicate paths excluded"
    queue.to_csv(QUEUE, index=False)
    exclusions.to_csv(EXCLUDED, index=False)

    counts = queue.groupby(["proposed_binary_class", "source", "original_category"], dropna=False).size()
    summary_rows = [{"class": key[0], "source": key[1], "original_category": key[2],
                     "selected": int(value), "target_per_class": target_per_class,
                     "seed": seed} for key, value in counts.items()]
    pd.DataFrame(summary_rows).to_csv(SUMMARY, index=False)
    status = {
        "queue_images": int(len(queue)), "target_per_class": int(target_per_class),
        "selected_by_class": {str(k): int(v) for k, v in queue.proposed_binary_class.value_counts().items()},
        "near_duplicate_candidates_excluded": int((exclusions.exclusion_reason == "unresolved_suspected_near_duplicate").sum()),
        "exact_conflict_candidates_excluded": int((exclusions.exclusion_reason == "unresolved_exact_label_conflict").sum()),
        "candidate_count": int(len(candidates)), "eligible_count": int(len(eligible)),
        "seed": int(seed),
        "note": "Selection is a review queue only. Source folders and AI suggestions are not human-verified labels.",
    }
    (PROCESSED / "pilot_review_queue_status.json").write_text(json.dumps(status, indent=2), encoding="utf-8")
    return queue, status


def main():
    parser = argparse.ArgumentParser(description="Prepare a stratified human review pilot queue")
    parser.add_argument("--per-class", type=int, default=300)
    parser.add_argument("--seed", type=int, default=SEED)
    args = parser.parse_args()
    if args.per_class < 1:
        parser.error("--per-class must be positive")
    _, status = build_pilot_queue(args.per_class, args.seed)
    print(json.dumps(status, indent=2))


if __name__ == "__main__":
    main()
