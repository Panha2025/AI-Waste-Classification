# Dataset label review status and protocol

## Current state

- `candidate_manifest.csv` contains 7,766 staged public image candidates as symlinks;
  the originals and source labels remain intact.
- The folder-derived provisional pool contains 7,227 recyclable candidates and 539
  residual (`trash`) candidates. These are not individually verified labels.
- Three exact duplicate groups (eight files) with source-label conflicts were excluded
  from the staged candidate pool. Their originals remain preserved.
- Fifty-seven suspected near-duplicate pairs remain review candidates. The detector
  output is not proof of duplication; no pair is automatically removed.
- A seeded pilot queue now contains 600 review candidates (300 per provisional class).
  It excludes 101 candidate images linked to unresolved near-duplicate pairs and keeps
  them in a separate exclusion log. All 600 remain unreviewed.
- No item has a human-verified binary label. No Cambodian local images are available.
- Therefore there are 0 verified labels and 7,766 provisional binary candidates; the
  current provisional count ratio is about 13.4:1. No final split manifest exists.

## Policy for reviewers

The target is commonly recyclable dry materials versus residual non-recyclable waste.
Cardboard, glass, metal, paper, and plastic are recyclable *candidates* only when the
visible item is sufficiently clean, dry, and recognizable. The source `trash` folder is
a residual candidate, subject to review. Batteries, biological waste, clothes, and
shoes remain out of this binary pool. An image cannot establish the actual acceptance
rules of a Cambodian collector or municipality; local policy requires separate
verification. Source-folder labels are never described as item-level ground truth.

Reviewers should choose `uncertain` if the object, material, condition, or relevant
context cannot be judged from the image. Difficult images should be retained with that
status rather than removed to improve a score. Exact conflicts stay quarantined until
resolved; similar-looking but distinct items stay distinct. Preserve provenance and
human notes for every decision.

## Local interface

Run `python -m waste_ai.pilot` once to create the deterministic queue, then run
`python -m waste_ai.label_review` from the project root and open
`http://127.0.0.1:8765`. It runs on loopback only by default and needs no new package.
The default view is the 600-image pilot; choose “All candidates” to browse beyond it.
The view supports keyboard shortcuts (arrows, 1/2/3/0), source/category context, and
explicit `recyclable`, `non_recyclable`, or `uncertain` decisions. Progress shows total
reviewed, binary approvals per class, uncertain, and pending. The duplicate
view presents the three conflicting exact groups and all 57 flagged pairs. Decisions
are appended to separate files in `data/processed/`; source images and original audit
files are not modified. Only explicit human binary decisions appear in
`human_verified_labels.csv`. This export does not itself create train/validation/test
splits.

The generated seed-42 queue has these provisional source/category counts:

| Proposed class | Source | Category | Selected |
|---|---|---|---:|
| Recyclable | Dataset 2 | Cardboard | 59 |
| Recyclable | Dataset 2 | Glass | 71 |
| Recyclable | Dataset 2 | Metal | 39 |
| Recyclable | Dataset 2 | Paper | 56 |
| Recyclable | Dataset 2 | Plastic | 67 |
| Recyclable | TrashNet | Cardboard | 1 |
| Recyclable | TrashNet | Glass | 4 |
| Recyclable | TrashNet | Metal | 1 |
| Recyclable | TrashNet | Paper | 1 |
| Recyclable | TrashNet | Plastic | 1 |
| Residual | Dataset 2 | Trash | 252 |
| Residual | TrashNet | Trash | 48 |

These are source-folder sampling counts, not confirmed labels. The small TrashNet
recyclable sample reflects its smaller share of the candidate pool; sampling preserves
approximate source/category proportions instead of equalizing the subcategories.

### AI suggestion status

The inspected project environment has no locally usable vision model, model weights,
Apple Vision Python bridge, or local model runner. No image-level AI suggestions were
generated. The interface displays that state instead of pretending source-folder
labels are model predictions. If a future *local* model emits suggestions, the optional
`data/processed/ai_label_suggestions.csv` format is:

```csv
id,ai_suggestion,confidence,reason
<candidate sha256>,recyclable,0.73,model's object/material cue
```

Allowed suggestions are `recyclable`, `non_recyclable`, or `uncertain`; confidence must
be treated as model confidence, not calibrated correctness or local-policy acceptance.
Imported suggestions remain unverified and cannot enter the verified export without a
human decision. This project currently has no such file or inference model.

## Review scale and split plan

A practical first pilot is 600 images: review a reproducible, source/category-stratified
sample of 300 provisional recyclable candidates and 300 provisional residual candidates.
Keep all unselected images and their provisional status. If the candidate pool permits,
the current 70/15/15 group split yields approximately 210/45/45 images per class, but
actual counts depend on confirmed labels and group sizes. Fewer than 300 individually
verified examples in either class will cause the audited pipeline to stop before writing
a split manifest. This is a pilot scale with a small test set, not a guarantee of
statistical sufficiency. It still needs Cambodian images for local evaluation. Matching
the *entire* present recyclable pool
would require 6,688 additional residual images if all current candidates were ultimately
approved. For broader coverage, collect residual examples across varied settings and
waste types instead of relying on one `trash` folder. These are planning targets, not
observed counts or a claim of statistical sufficiency.

Before splitting, resolve/quarantine near-duplicate relationships and use connected
duplicate/session groups as indivisible units. Verify both classes and sufficient
independent groups in every split. If a class or split is too small, collect more data
instead of synthesizing evaluation examples. Any augmentation or sampling is training
only. Report macro F1, class-specific precision/recall, balanced accuracy, confusion
matrices, class counts, and both combined and independent local test results.

Optional imbalance handling is implemented as `--class-weight balanced`; weights are
computed from training labels only. It remains opt-in. No split manifest or training run
was created while labels remain unreviewed; bot work was not performed.
