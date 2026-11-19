# Professor Pin Kuntha's Feedback → Implementation and Final Evidence

The original Group 4 proposal was graded 21.2/25. The professor accepted the overall
research direction but requested clearer evidence. Check off only after producing proof:

- [ ] **Dataset size & hardware** — run `python -m waste_ai.data`; include
  `data/processed/source_summary.csv`. After actual training, include CPU, GPU,
  RAM, Python, TF, execution time from `artifacts/metrics/*_train.json`.
- [ ] **Experimental setup** — include `split_manifest.csv`, split_counts,
  method and seed; document 70/15/15 targets and group-based actual distribution.
  Use `test` for comparable model evaluation and **separate** `local_test` for
  generalization. Never contaminate test splits with training/augmentation.
- [ ] **Team contributions** — have all five members fill factual work completed
  in `docs/TEAM_CONTRIBUTIONS.md` and explain methodology during presentation.
- [ ] **Dataset visualization** — include `data/processed/dataset_examples.png`
  plus higher-resolution examples from *every* dataset if the automatic grid
  reaches its 6-source limit.
- [ ] **Proper methodology flowchart** — render diagram in `docs/METHODOLOGY.md`.
- [ ] **Cleaning and selection** — present `cleaning_log.csv` with exclusion
  criteria and counts. Audit random samples; never remove difficult images just
  to inflate reported accuracy. Review potentially wrong labels manually.
- [ ] **Local dataset** — justify collection in Cambodia, capture different
  backgrounds and lighting, preserve capture sessions, and report independent
  `local_test` performance separately. Explain any gap compared with combined test.

## Suggested slides for defense

1. Background and objective
2. Research gap and why Cambodian local images matter
3. Data sources and exact counts
4. Cleaning examples and label policy
5. Leakage-safe experimental design and methodology flowchart
6. Three architectures and hardware/training settings
7. Validation / test comparison table
8. Confusion matrices and errors
9. Local test evaluation and limitations
10. Telegram photo upload → prediction live demo
11. Real member contributions
12. Conclusion and future work

## Evidence NOT yet available in initial ZIP

- Actual source data and any locally collected photos
- Verified data cleaning decisions / source licensing
- Training hardware measured during training
- Model weights, trained accuracies, confusion matrices
- Independent local evaluation metrics
- Successful live bot-to-Telegram message exchange

**Never replace missing results with estimates or invented scores.**
