"""Dataset audit, exact-duplicate handling, group-aware splits, and visualization.

No datasets are bundled. Every resulting metric reflects user's real data only.
"""
from __future__ import annotations
import argparse
from collections import Counter, defaultdict
from datetime import datetime, timezone
import hashlib
import json
import random
from pathlib import Path

from PIL import Image, ImageOps, UnidentifiedImageError
import pandas as pd

from waste_ai.config import (RAW_DATA, PROCESSED, LABEL_MAP, ALLOWED_SUFFIXES,
                             MAX_IMAGE_BYTES, MAX_IMAGE_PIXELS, SEED, LABELS, ensure_dirs)


class DatasetError(ValueError):
    pass


def dhash(image: Image.Image) -> int:
    """64-bit perceptual difference hash. Helps detect simple resized image duplicates."""
    grayscale = ImageOps.exif_transpose(image).convert("L").resize((9, 8), Image.Resampling.LANCZOS)
    pixels = list(grayscale.tobytes())
    value = 0
    for row in range(8):
        for col in range(8):
            value = (value << 1) | int(pixels[row * 9 + col] > pixels[row * 9 + col + 1])
    return value


def image_info(path: Path):
    if path.stat().st_size > MAX_IMAGE_BYTES:
        raise DatasetError("image_over_10mb")
    try:
        with Image.open(path) as img:
            width, height = img.size
            if width * height > MAX_IMAGE_PIXELS:
                raise DatasetError("image_over_20_megapixels")
            img.load()
            p_hash = dhash(img)
            sha = hashlib.sha256(path.read_bytes()).hexdigest()
            return {"width": width, "height": height, "sha256": sha, "dhash": f"{p_hash:016x}"}
    except (OSError, ValueError, UnidentifiedImageError) as e:
        if isinstance(e, DatasetError):
            raise
        raise DatasetError("corrupt_or_unreadable") from e


def image_source(path: Path):
    rel = path.relative_to(RAW_DATA)
    parts = rel.parts
    if parts[0] == "public" and len(parts) >= 4:
        source = f"public/{parts[1]}"
        raw_label = parts[2].lower()
        group = f"public:{parts[1]}/{parts[2]}/{path.stem}"
    elif parts[0] == "local" and len(parts) >= 3:
        source = "local"
        raw_label = parts[1].lower()
        # Group photos from the same session so variants cannot cross train/test.
        group = (f"local:{parts[1]}/{parts[2]}" if len(parts) >= 4
                 else f"local:{parts[1]}/{path.stem}")
    else:
        return None
    return source, raw_label, group


def mapped_label(mapping, source, raw_label):
    value = mapping.get("sources", {}).get(source, {}).get(raw_label)
    if value is None:
        value = mapping.get("global", {}).get(raw_label)
    return value if value in LABELS else None


def near_same_image(path_a: str, path_b: Path, max_mean_difference=8.0) -> bool:
    """Conservative pixel check in addition to perceptual hash.

    dHash alone can collide for visually different flat-color backgrounds. Require
    close RGB appearance, too, before treating a pair as a near duplicate.
    """
    import numpy as np
    with Image.open(path_a) as first, Image.open(path_b) as second:
        a = np.asarray(ImageOps.exif_transpose(first).convert("RGB").resize((32, 32)), dtype=np.float32)
        b = np.asarray(ImageOps.exif_transpose(second).convert("RGB").resize((32, 32)), dtype=np.float32)
        return float(np.abs(a - b).mean()) <= max_mean_difference


def split_groups_by_label(rows, names, seed=SEED, fractions=(0.70, 0.15, 0.15)):
    """Assign entire capture groups within each class; avoids related-image leakage.

    Group-level proportions are approximate; generated summary provides actual counts.
    """
    rng = random.Random(seed)
    grouped = defaultdict(lambda: defaultdict(list))
    for r in rows:
        grouped[r["label"]][r["capture_group"]].append(r)
    assignments = {}
    for label in LABELS:
        if label not in grouped:
            continue
        groups = list(grouped[label])
        if len(groups) < len(names):
            raise DatasetError(
                f"Need >= {len(names)} independent capture groups for class '{label}' "
                f"to create {names}; found {len(groups)}. Collect more images/sessions."
            )
        rng.shuffle(groups)
        # Greedy approximately weighted assignment while guaranteeing non-empty splits.
        targets = [fractions[i] * sum(len(v) for v in grouped[label].values())
                   for i in range(len(names))]
        tally = [0] * len(names)
        for idx, group in enumerate(groups):
            remaining = len(groups) - idx
            empty = [i for i, n in enumerate(tally) if n == 0]
            if remaining == len(empty):
                chosen = empty[0]
            else:
                chosen = max(range(len(names)), key=lambda i: targets[i] - tally[i])
            assignments[group] = names[chosen]
            tally[chosen] += len(grouped[label][group])
    return assignments


def save_visualization(rows, out_file, max_sources=6, images_per_source=3):
    if not rows:
        return
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    by_source = defaultdict(list)
    for r in rows:
        by_source[r["source"]].append(r)
    sources = sorted(by_source)[:max_sources]
    cols = images_per_source
    fig, axes = plt.subplots(len(sources), cols, figsize=(cols * 3.1, len(sources) * 3.0), squeeze=False)
    for i, src in enumerate(sources):
        # Show both labels where present; prefer diversity over a single class.
        buckets = {label: [r for r in by_source[src] if r["label"] == label] for label in LABELS}
        examples = []
        while len(examples) < cols and any(buckets.values()):
            for label in LABELS:
                if buckets[label] and len(examples) < cols:
                    examples.append(buckets[label].pop(0))
        for j in range(cols):
            ax = axes[i][j]
            ax.axis("off")
            if j >= len(examples):
                continue
            r = examples[j]
            with Image.open(r["path"]) as im:
                ax.imshow(ImageOps.exif_transpose(im).convert("RGB"))
            ax.set_title(f"{src} | {r['label']}", fontsize=8)
    fig.suptitle("Representative waste images by dataset source", fontsize=13)
    fig.tight_layout()
    out_file.parent.mkdir(parents=True, exist_ok=True)
    fig.savefig(out_file, dpi=125)
    plt.close(fig)


def save_class_distribution(rows, csv_file, chart_file):
    counts = Counter(r["label"] for r in rows)
    frame = pd.DataFrame([{"label": label, "images": int(counts[label])} for label in LABELS])
    frame.to_csv(csv_file, index=False)
    import matplotlib
    matplotlib.use("Agg")
    import matplotlib.pyplot as plt
    fig, ax = plt.subplots(figsize=(8, 4.5))
    ax.bar(frame.label, frame.images, color="#28745a")
    ax.set_ylabel("Images")
    ax.set_title("Public dataset images by source category")
    ax.tick_params(axis="x", rotation=25)
    fig.tight_layout()
    fig.savefig(chart_file, dpi=150)
    plt.close(fig)


def prepare(seed=SEED, local_fraction=0.20):
    ensure_dirs()
    if not 0 <= local_fraction <= 0.5:
        raise DatasetError("local_fraction must be from 0 to 0.5")
    mapping = json.loads(LABEL_MAP.read_text(encoding="utf8"))
    candidates = sorted(p for p in RAW_DATA.rglob("*") if p.is_file() and not p.name.startswith("."))
    if not candidates:
        raise DatasetError(f"No images found. See README: place datasets under {RAW_DATA}")

    rows, cleaning, source_originals = [], [], Counter()
    seen_sha = {}
    # Near-duplicate flags from the prior audit are held out until adjudicated.
    # Confirmed same-image/object pairs share a capture group; distinct pairs stay
    # separate. This uses source paths from the candidate manifest, never pixel guesses.
    staged_by_original = {}
    candidate_file = PROCESSED / "candidate_manifest.csv"
    if candidate_file.exists():
        candidates_df = pd.read_csv(candidate_file).fillna("")
        for _, item in candidates_df.iterrows():
            staged = str((Path(item["staged_path"]).resolve()))
            original = str(item["original_path"])
            staged_by_original[original] = staged
    duplicate_decisions = {}
    duplicate_review_file = PROCESSED / "human_duplicate_reviews.csv"
    if duplicate_review_file.exists():
        for _, item in pd.read_csv(duplicate_review_file).fillna("").iterrows():
            duplicate_decisions[str(item.get("id", ""))] = str(item.get("decision", ""))
    near_groups, quarantine_staged = {}, set()
    near_graph, unresolved_nodes = defaultdict(set), set()
    near_file = PROCESSED / "near_duplicate_review.csv"
    if near_file.exists():
        for index, item in pd.read_csv(near_file).fillna("").iterrows():
            review_id = f"near-{index}"
            left_original, right_original = str(item["path"]), str(item["candidate_duplicate_of"])
            left = staged_by_original.get(left_original)
            right = staged_by_original.get(right_original)
            decision = duplicate_decisions.get(review_id, "")
            if decision != "distinct":
                nodes = [node for node in (left, right) if node]
                if decision in ("same_image", "same_object") and len(nodes) == 2:
                    near_graph[nodes[0]].add(nodes[1])
                    near_graph[nodes[1]].add(nodes[0])
                else:
                    for node in nodes:
                        near_graph[node].add(node)
                if decision not in ("same_image", "same_object"):
                    unresolved_nodes.update(nodes)
    visited = set()
    for start in near_graph:
        if start in visited:
            continue
        stack, component = [start], set()
        while stack:
            current = stack.pop()
            if current in component:
                continue
            component.add(current)
            stack.extend(near_graph[current] - component)
        visited.update(component)
        if component & unresolved_nodes:
            quarantine_staged.update(component)
        else:
            group = "reviewed_duplicate:" + min(component)
            for staged in component:
                near_groups[staged] = group
    # Public first so local test does not contain identical public-training examples.
    ordered = sorted(candidates, key=lambda p: (not str(p.relative_to(RAW_DATA)).startswith("public/"), str(p)))
    for path in ordered:
        info = image_source(path)
        if info is None:
            cleaning.append({"path": str(path), "source": "unrecognized", "reason": "unexpected_folder_layout"})
            continue
        source, raw_label, capture_group = info
        source_originals[source] += 1
        resolved_path = str(path.resolve())
        if resolved_path in quarantine_staged:
            cleaning.append({"path": str(path), "source": source,
                             "reason": "near_duplicate_review_pending_or_uncertain"})
            continue
        capture_group = near_groups.get(resolved_path, capture_group)
        reason = None
        if path.suffix.lower() not in ALLOWED_SUFFIXES:
            reason = "unsupported_file_extension"
        label = mapped_label(mapping, source, raw_label)
        if label is None:
            reason = reason or "unmapped_or_unverified_label"
        if reason:
            cleaning.append({"path": str(path), "source": source, "reason": reason})
            continue
        try:
            image_data = image_info(path)
        except DatasetError as e:
            cleaning.append({"path": str(path), "source": source, "reason": str(e)})
            continue
        existing_sha = seen_sha.get(image_data["sha256"])
        comparison = [(existing_sha, "exact_duplicate")] if existing_sha is not None else []
        if comparison:
            idx, reason = comparison[0]
            kept = rows[idx]
            if kept["label"] != label:
                conflict = PROCESSED / "label_conflicts.csv"
                pd.DataFrame([{
                    "kept_path": kept["path"], "kept_label": kept["label"],
                    "conflicting_path": str(path), "conflicting_label": label,
                    "reason": reason,
                }]).to_csv(conflict, index=False)
                raise DatasetError(
                    f"Conflicting labels for {path} and {kept['path']}. "
                    f"Resolve them manually before continuing. See {conflict}"
                )
            cleaning.append({"path": str(path), "source": source,
                             "reason": reason, "duplicate_of": kept["path"]})
            continue
        record = {"path": str(path.resolve()), "source": source, "raw_label": raw_label,
                  "label": label, "capture_group": capture_group, **image_data}
        idx = len(rows)
        rows.append(record)
        seen_sha[image_data["sha256"]] = idx

    pd.DataFrame(cleaning, columns=["path", "source", "reason", "duplicate_of"]).to_csv(
        PROCESSED / "cleaning_log.csv", index=False)
    if not rows:
        raise DatasetError("No valid labelled images remain. Check label_map.json and cleaning_log.csv")
    local = [r for r in rows if r["source"] == "local"]
    local_assign = {}
    warnings = []
    if local and local_fraction > 0:
        # Independent local-only holdout, separate from the combined dataset test.
        grouped = defaultdict(lambda: defaultdict(list))
        for r in local:
            grouped[r["label"]][r["capture_group"]].append(r)
        rng = random.Random(seed + 7919)
        for label in LABELS:
            groups = list(grouped[label])
            if len(groups) < 2:
                warnings.append(f"No independent local holdout for {label}: need 2+ capture sessions")
                continue
            rng.shuffle(groups)
            target = max(1, min(len(groups)-1, round(len(groups) * local_fraction)))
            for g in groups[:target]:
                local_assign[g] = "local_test"
    elif not local:
        warnings.append("No local images. Independent Cambodian local evaluation is not yet possible.")
    remainder = [r for r in rows if r["capture_group"] not in local_assign]
    assignments = split_groups_by_label(remainder, ("train", "val", "test"), seed)
    assignments.update(local_assign)
    for r in rows:
        r["split"] = assignments[r["capture_group"]]
    df = pd.DataFrame(rows)
    df.to_csv(PROCESSED / "split_manifest.csv", index=False)
    counts = df.groupby(["source", "label", "split"]).size().rename("images").reset_index()
    counts.to_csv(PROCESSED / "split_counts.csv", index=False)
    used_counts = Counter(r["source"] for r in rows)
    removed_counts = Counter(r["source"] for r in cleaning)
    table = [{"source": src, "original": int(original), "removed": int(removed_counts[src]),
              "used": int(used_counts[src])} for src, original in sorted(source_originals.items())]
    pd.DataFrame(table).to_csv(PROCESSED / "source_summary.csv", index=False)
    # One example for each active source category enables a quick label spot-check.
    save_visualization(rows, PROCESSED / "dataset_examples.png",
                       images_per_source=len(LABELS))
    save_class_distribution(rows, PROCESSED / "class_distribution.csv",
                            PROCESSED / "class_distribution.png")
    summary = {"created_utc": datetime.now(timezone.utc).isoformat(), "seed": seed,
               "local_holdout_fraction_target": local_fraction,
               "total_original_images": sum(source_originals.values()),
               "total_removed_images": len(cleaning), "total_used_images": len(rows),
               "actual_split_counts": {str(k): int(v) for k, v in df["split"].value_counts().items()},
               "class_counts": {str(k): int(v) for k, v in df["label"].value_counts().items()},
               "source_counts": table, "warnings": warnings,
               "important": "Labels retain source-provided dataset categories; they are not human-verified local recycling acceptance decisions."}
    (PROCESSED / "summary.json").write_text(json.dumps(summary, indent=2), encoding="utf8")
    print(json.dumps(summary, indent=2))
    return summary


def main():
    p = argparse.ArgumentParser(description="Clean, audit, group-split, and visualize waste image data")
    p.add_argument("--seed", type=int, default=SEED)
    p.add_argument("--local-fraction", type=float, default=0.20)
    args = p.parse_args()
    try:
        prepare(args.seed, args.local_fraction)
    except DatasetError as e:
        p.error(str(e))


if __name__ == "__main__":
    main()
