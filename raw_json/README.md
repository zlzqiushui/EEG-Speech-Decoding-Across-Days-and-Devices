# raw_json — source result JSONs used by the statistics

This folder contains **only the JSON result files that the paper's statistics are
computed from** — no model weights, no checkpoints, no logs, no training curves.

## What was taken

| Module | Source directory pattern | Original file type | Results in release |
|---|---|---|---|
| A — intra-domain DG | `result_v4/models` | `test_results.json` | 1,500 |
| B — SDA | `finetune_result_v3/models` | `results.json` | 4,725 |

Selection criteria — exactly the same filters `code/analysis/analyze_paper.py` uses:

- features: **`envelope`, `mel_10`, `wav2vecbase9`** (the 80-D `mel` is excluded),
- seed: **42**,
- Module A domains: `day1`, `day2`, `brk`, `neurascan`; ratios r = 0.2 … 1.0,
- Module B transfers: `day1_to_day2`, `day2_to_day1`, `neurascan_to_brk`;
  strategies S1–S4 at r = 0.2 … 1.0, and S5 only at r = 1.0.

The 6,225 retained result values are merged into **21 files**, one per
(module × feature × condition).

## Layout

```
raw_json/
├── module_A_intra_domain/
│   ├── envelope/     day1.json, day2.json, brk.json, neurascan.json
│   ├── mel_10/       same four files
│   └── wav2vecbase9/ same four files
└── module_B_sda/
    ├── envelope/     day1_to_day2.json, day2_to_day1.json, neurascan_to_brk.json
    ├── mel_10/       same three files
    └── wav2vecbase9/ same three files
```

## File format

```jsonc
{
  "module": "A_intra_domain",          // or "B_sda"
  "feature": "mel_10",
  "condition": "neurascan",            // domain (A) or transfer (B)
  "seed": 42,
  "metric": "top-1 match accuracy (1 true stimulus vs 4 random distractors, i.e. 5-candidate)",
  "chance_pct": 20.0,
  "value_unit": "percent",
  "n_subjects": 25,
  "strategies": { "S1": "Zero-shot Historical Model", ... },   // Module B only
  "source_root": "result_v4/models",
  "source_name_template": "BrainNetworkCL-...-valid1-mel_10-sub{N}-ratio{R}-neurascan-seed42/test_results.json",
  "note": "data_raw = original 'test_accuracy' field (fraction); data = data_raw * 100, rounded to 2 dp",

  "data": {                            // percent, 2 dp — identical to raw_results/*.csv
    "sub-01": { "r0.2": 41.28, "r0.4": 42.30, "r0.6": 46.63, "r0.8": 51.56, "r1.0": 56.97 }
  },
  "data_raw": {                        // test_accuracy fractions
    "sub-01": { "r0.2": 0.4127906976744186, ... }
  }
}
```

Module B keys are `S{1..4}_r{0.2..1.0}` and `S5_r1.0`, e.g. `"S2_r0.2": 51.90`.

All 6,225 retained cells are populated. The `wav2vecbase9/neurascan` file includes
complete sub-24 and sub-25 rows, with `data_raw` fractions and the corresponding
two-decimal `data` percentages.

## Provenance

The consolidated records identify the source run directory pattern:

```
result_v4/models/BrainNetworkCL-bs32-sl5-ks3-dor0.5-att256-nn32-valid1-<feature>-sub<N>-ratio<R>-<domain>-seed42/test_results.json
finetune_result_v3/models/BrainNetworkCL-bs32-sl5-ks3-dor0.5-att256-nn32-valid1-<feature>-sub<N>-ratio<R>-finetune-<transfer>-S<S>-seed42/results.json
```

The underlying run files and checkpoints are not bundled in this repository;
the released JSON contains their reported test accuracies.

## Regenerate

```bash
# requires the original project root containing result_v4/ and finetune_result_v3/
python code/analysis/export_raw_json.py
# Re-export retains any completed values already present in raw_json/ when
# their individual run files are unavailable.
```

## Cross-check

All 6,225 cells in `raw_results/*.csv` are compared against the matching
percentages here — **0 mismatches**. Statistical reproduction from these
released values is described in `code/analysis/README.md`.
