# EEG Speech Decoding Across Days and Devices

Code and participant-level results for *Evaluating Generalization Performance
and Domain Adaptation Strategies for EEG-Based Speech Decoding Across Days and
Devices*.

The study uses PKUEEG recordings from 25 participants across three days to
examine cross-day and cross-device EEG–speech decoding and recalibration. The
five-candidate match–mismatch task has a 20% chance level.

## Contents

- `code/module_a_dg/`: generalization experiments.
- `code/module_b_sda/`: adaptation strategies S1–S5.
- `code/analysis/`: statistics and figures.
- `raw_json/`: source accuracy values.
- `raw_results/`: participant-level CSV tables.

## Reproduce the analysis

```bash
python -m pip install -r code/analysis/requirements.txt
python code/analysis/analyze_paper.py
```

The analysis uses the bundled results and writes to `code/analysis/output/`.
EEG recordings and model checkpoints are not included. Results use seed 42;
S5 is included only at the full target-data ratio (`r=1.0`).
