# Aligned A/C/D Cube O50 paired-result archive

This directory is the machine-readable archive for the fixed-training-seed
Aligned world/tail ablation on OGBench Cube. It contains six matched planning
selections (seeds 42--47), 50 start--goal pairs per selection, and 300 paired
episodes in total.

## Factorial labels

- **A:** original LeWM world model with the original terminal cost.
- **B:** original LeWM world model with a matched boundary-anchored tail trained
  in the original latent coordinates. **Not completed.**
- **B-prime:** original LeWM world model with the tail trained in Aligned latent
  coordinates. This is an invalid cross-coordinate diagnostic, not cell B.
- **C:** Aligned world model with the original terminal cost.
- **D:** Aligned world model with its matched boundary-anchored MC tail.

All A/C/D rows use training seed 3072. Seeds 42--47 are planning-selection
seeds, not independently trained models.

## Files

- `paired_outcomes.csv`: the 300 matched pairs and A/C/D success booleans.
- `summary.json`: pooled and per-selection rates, exact McNemar tables,
  Holm-adjusted p-values, direction counts, and sample-overlap audit.
- `selection_manifests.json`: exact selections and per-run hashes, checkpoint
  identifiers, runtime versions, GPU, elapsed time, Git revision, and source
  paths.
- `auxiliary_diagnostics.json`: B-prime and the existing unanchored MC-head
  outcomes, kept separate from the clean A/C/D analysis.
- `planner_diagnostics.json`: the complete 134 KB O5 CEM diagnostic trace.
- `planner_diagnostics_summary.json`: the compact mechanism summary derived
  from that trace.
- `checksums.sha256`: SHA-256 values for the committed files in this directory.

No dataset rows, images, state vectors, logs, or checkpoint tensors are stored
here.

## Recompute and verify

From the repository root:

```bash
python scripts/summarize_aligned_acd_o50.py --check
shasum -a 256 -c reports/artifacts/aligned_acd_o50_seed3072/checksums.sha256
```

The summarizer uses only the Python standard library. `pair_hash` is SHA-256 of
compact, key-sorted JSON containing `episode_index`, `goal_step`, and
`start_step`. `selection_hash` is SHA-256 of the exact
`episode_selection.json` bytes written by the evaluator.

## Sample audit

- 300/300 start--goal triples have unique `pair_hash` values.
- No exact pair occurs in more than one planning selection.
- The pairs come from 296 unique source episodes.
- Source episodes 5132, 7002, 8400, and 9267 each occur twice, but with
  different start--goal steps.

## Provenance limitation

The evaluators persisted the full Git revision but did not persist a run-time
dirty-worktree boolean. `selection_manifests.json` therefore records
`dirty_worktree: null` and the missing-field reason rather than inferring a
value retrospectively. This limitation is separate from the result-file,
selection, protocol-manifest, and checkpoint SHA-256 values, all of which are
recorded.

The planned vectorized O200 run exceeded the RTX 3090 host's approximately
15 GiB memory budget and was terminated by the operating system. This archive
is six matched O50 executions totalling 300 episodes; it is not one O200 run.
