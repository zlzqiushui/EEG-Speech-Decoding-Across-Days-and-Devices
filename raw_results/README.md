# raw_results — per-subject accuracy tables

Per-subject top-1 match accuracy for **all 25 subjects**, organised by
**module × feature**. Generated from the released `raw_json/` by
`code/analysis/extract_raw_results.py`.

## Conventions

- **Metric**: top-1 match accuracy (1 true stimulus vs 4 random distractors,
  **chance = 20 %**), given in **percent** with 2 decimals.
- **Rows**: one per subject (`sub-01` … `sub-25`).
- **Seed**: 42 throughout — the only seed used in the paper.
- All 6,225 retained result cells are populated, including complete sub-24 and sub-25
  rows in `wav2vecbase9 / neurascan`.

## Layout

```
raw_results/
├── module_A_intra_domain/          # Module A — train and test in the same domain
│   ├── envelope/     day1.csv, day2.csv, brk.csv, neurascan.csv
│   ├── mel_10/       same four files   (10-D mel, called "Mel" in the paper)
│   └── wav2vecbase9/ same four files
└── module_B_sda/                   # Module B — 5 SDA strategies
    ├── envelope/     day1_to_day2.csv, day2_to_day1.csv, neurascan_to_brk.csv
    ├── mel_10/       same three files
    └── wav2vecbase9/ same three files
```

The 80-D `mel` feature is deliberately **not** included.

## CSV format

**Module A** — `{domain}.csv`, columns are the training-data ratio *r*:

```
subject, r0.2,  r0.4,  r0.6,  r0.8,  r1.0
sub-01,  36.97, 34.45, 42.02, 42.02, 47.06
...
```

**Module B** — `{transfer}.csv`, with S1–S4 at five ratios and S5 only at r=1.0 (21 columns):

```
subject, S1_r0.2, S1_r0.4, ..., S4_r1.0, S5_r1.0
sub-01,  20.35,   20.35,   ..., 18.58,   21.24
...
```

Transfer settings: `day1_to_day2` / `day2_to_day1` (cross-day, same Neuroscan
device), `neurascan_to_brk` (cross-device, Neuroscan → Neuracle).

## How these map to the paper

- `module_A_intra_domain/*.csv` → target-trained references and the fixed
  Day-1 source accuracy on Day 2 (`neurascan/r1.0`).
- `module_B_sda/neurascan_to_brk.csv` → fixed-source Day-3 accuracy
  (`S1_r1.0`); both sources form current-paper Table 1.
- `module_B_sda/day*_to_day*.csv` → bidirectional cross-day analyses; the
  two directions are averaged within participant.
- Aggregated statistics (mean ± SEM, paired *t*-tests) →
  `code/analysis/output/statistics/`.

## Regenerate

```bash
python code/analysis/extract_raw_results.py
```

The same values and their full-precision `test_accuracy` fractions are in
`raw_json/`.
