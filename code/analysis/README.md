# Reproduce the current manuscript analysis

From the repository root, on Python 3.9 or newer:

```bash
python -m pip install -r code/analysis/requirements.txt
python code/analysis/analyze_paper.py
```

This CPU analysis needs only the bundled `raw_json/` and four Python packages.
It does not need EEG recordings, model checkpoints, PyTorch, or the original
cluster directory. Paths are resolved relative to the script, so the command
also works from another working directory. Use `--output /path/to/results` to
choose an output directory, or `--raw-json /path/to/raw_json` to select inputs.

## Inputs and precision

The extractor reads **`data_raw`** (accuracy fractions), multiplies by 100,
and checks that every value rounds to the JSON `data` percentage. Statistics
and figures are now computed from the **full-precision fractions**. The
two-decimal participant CSVs remain available for comparison with earlier
paper exports; they are not used for the updated statistical tests.

There are 21 JSON files and 6,225 observed values. Module B retains S5 only
at r=1.0. Module A
`wav2vecbase9/neurascan` includes complete sub-24 and sub-25 results at
full `test_accuracy` fraction precision. Unexpected missing cells, invalid
metadata, mismatched percentage fields, and incomplete cross-day pairs cause
the run to fail.

## Analysis reproduced

- Participant is the inferential unit. Average the two cross-day directions
  **within each participant** before computing means, SEMs, or tests.
- Zero-shot vs. 20% chance: six two-sided one-sample t tests.
- Target-trained minus zero-shot loss: six paired comparisons, using Module A
  Day 1/Day 2 references for cross-day and `brk` for replacement.
- S2 minus S1 at r=.2: six paired comparisons.
- S2 minus S3 at r=.2 and r=1: twelve paired comparisons.
- Fixed Day-1 source: three paired Day-2 minus Day-3 comparisons, using
  Module A `neurascan/r1.0` and Module B `neurascan_to_brk/S1_r1.0`.
  All three features use **25 complete participant pairs**.
- Holm correction is applied separately within each of these five families.
  Confidence intervals are unadjusted 95% t intervals using n−1 degrees of freedom.
- Figure 3 shows mean ± SEM and paired loss with 95% CI. Figure 4 shows
  S2/S3/S4 at all five ratios, S1 as a zero-shot line and S5 only at r=1.
  S5 is not a sparse-budget curve because its training merge bypassed the
  target-ratio selection.

This reproduces the statistics and empirical figures of the current v6 paper.
It does not rerun training, verify checkpoint identities, or regenerate the
conceptual non-stationarity/preprocessing illustrations (Figures 1 and 2).

## Outputs and verification

`output/` contains:

- `raw_results/`: the 21 two-decimal participant CSVs (generated locally).
- `raw_results_full/`: full-precision participant tables used for analysis
  (generated locally and ignored by Git).
- `statistics/`: all 33 tests, fixed-source participant pairs, 102 Figure-4
  mean/SEM rows, Figure-3 estimates, and direction-specific descriptive results.
  `statistics/rounded_export/` retains the five test families calculated from
  the historical two-decimal CSV export for comparison.
- `figures/fig3_generalization.{pdf,png}` and `fig4_calibration.{pdf,png}`.
- `reproduction_report.json`: pass/fail, input SHA-256 hashes, missing-cell audit,
  package versions, and checks against the saved manuscript results.

The six files in `reference/` are the saved **current full-precision results**,
not outputs substituted into the computation. They are read only at the final
comparison stage. Verification checks all columns of 33 test rows
and 102 calibration rows with absolute and relative tolerance 1e-10. When the
bundled `raw_results/` is present beside `raw_json/`, all 21 reconstructed CSVs
are additionally compared with it exactly.

The rounding audit independently recalculates the earlier two-decimal export.
Its largest numeric difference from the full-precision results is below 0.005,
and Holm decisions are unchanged. Two one-decimal displays differ: the Mel-10
replacement loss is 16.8 instead of 16.9 pp, and the fixed-source wav2vec
upper CI is 9.0 instead of 8.9 pp. Full-precision fixed-source wav2vec means
are 46.28632% (Day 2) and 42.477876% (Day 3), with a 3.808444 pp difference.

On the released data, all checks pass. Fixed-source Holm p-values are
1.000, 1.000, and 0.418516; none establishes a significant difference or
statistical equivalence between continued use and replacement.

## Code provenance

This replaces the earlier repository analysis with the code used locally for
the current manuscript, with portable paths and a JSON input adapter:

| File | Local manuscript source / changes |
|---|---|
| `reanalyze_submission.py` | Statistical functions from `paper_submission/code/analysis/reanalyze_submission.py`; old plotting functions removed, paths redirected, missing-data guards added. |
| `make_figures.py` | Empirical plotting section of `paper_submission/output/pdf/scripts/build_figures.py`; wrapped as a function and output paths redirected. |
| `fixed_source.py` | Fixed-source calculation from `v6_analysis/build_v6.py`; manuscript editing removed and participant provenance paths made relative. |
| `extract_raw_results.py` | New adapter for the merged JSON release; preserves the original CSV extraction precision. |
| `analyze_paper.py` | New orchestration and regression verification entry point. |
| `export_raw_json.py` | Retained original exporter; requires the original cluster result directories and is **not** needed to reproduce the paper from this release. |

The six current reference tables were updated after independent calculations
with the full-precision `data_raw` values. Earlier two-decimal export results
remain available in `output/statistics/rounded_export/`.
