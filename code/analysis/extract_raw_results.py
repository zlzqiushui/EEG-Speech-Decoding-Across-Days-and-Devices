#!/usr/bin/env python3
"""Rebuild participant CSVs from bundled data_raw fractions, using paper precision."""
import argparse
import csv
import hashlib
import json
import math
from pathlib import Path

FEATURES = ('envelope', 'mel_10', 'wav2vecbase9')
RATIOS = (.2, .4, .6, .8, 1.)
SUBJECTS = [f'sub-{n:02d}' for n in range(1, 26)]
CONDITIONS = {
    'module_A_intra_domain': ('day1', 'day2', 'brk', 'neurascan'),
    'module_B_sda': ('day1_to_day2', 'day2_to_day1', 'neurascan_to_brk'),
}


def extract(raw_json, output, round_to_export=True):
    raw_json, output = Path(raw_json).resolve(), Path(output).resolve()
    if output == raw_json or raw_json in output.parents:
        raise ValueError('Output must be outside raw_json.')
    audit = {'files': [], 'cells': 0, 'present': 0, 'missing': 0,
             'precision': ('data_raw * 100, formatted to 2 decimal places (percentage points)'
                           if round_to_export else 'data_raw * 100, without decimal rounding')}
    for module, conditions in CONDITIONS.items():
        columns = ([f'r{r:.1f}' for r in RATIOS] if module.startswith('module_A') else
                   [f'S{s}_r{r:.1f}' for s in range(1, 6)
                    for r in ((1.,) if s == 5 else RATIOS)])
        for feature in FEATURES:
            for condition in conditions:
                rel = Path(module) / feature / f'{condition}.json'
                source = raw_json / rel
                obj = json.loads(source.read_text())
                expected_module = 'A_intra_domain' if module.startswith('module_A') else 'B_sda'
                if (obj['seed'], obj['value_unit'], obj['feature'], obj['condition'], obj['module'], obj['chance_pct']) != (42, 'percent', feature, condition, expected_module, 20.):
                    raise ValueError(f'Unexpected metadata in {rel}')
                if set(obj['data']) != set(obj['data_raw']) or not set(obj['data_raw']).issubset(SUBJECTS):
                    raise ValueError(f'Unexpected subject IDs in {rel}')
                if obj['n_subjects'] != len(obj['data_raw']):
                    raise ValueError(f'Incorrect subject count in {rel}')
                rows, missing = [], []
                for sub in SUBJECTS:
                    if sub in obj['data_raw']:
                        if set(obj['data_raw'][sub]) != set(columns) or set(obj['data'][sub]) != set(columns):
                            raise ValueError(f'Unexpected result columns: {rel}/{sub}')
                    row = [sub]
                    for column in columns:
                        raw = obj['data_raw'].get(sub, {}).get(column)
                        pct = obj['data'].get(sub, {}).get(column)
                        if raw is not None and (not isinstance(raw, (int, float)) or not math.isfinite(raw) or not 0 <= raw <= 1):
                            raise ValueError(f'Invalid accuracy: {rel}/{sub}/{column}')
                        value = None if raw is None else raw * 100
                        if value is not None and float(f'{value:.2f}') != pct:
                            raise ValueError(f'data_raw/data mismatch: {rel}/{sub}/{column}')
                        if value is None:
                            missing.append(f'{sub}/{column}')
                        row.append('' if value is None else
                                   f'{value:.2f}' if round_to_export else repr(value))
                    rows.append(row)
                if missing:
                    raise ValueError(f'Unexpected missing results in {rel}: {missing}')
                dest = output / rel.with_suffix('.csv')
                dest.parent.mkdir(parents=True, exist_ok=True)
                with dest.open('w', newline='') as handle:
                    writer = csv.writer(handle, lineterminator='\n')
                    writer.writerow(['subject'] + columns)
                    writer.writerows(rows)
                cells = len(SUBJECTS) * len(columns)
                audit['cells'] += cells
                audit['present'] += cells - len(missing)
                audit['missing'] += len(missing)
                audit['files'].append({'path': rel.as_posix(), 'sha256': hashlib.sha256(source.read_bytes()).hexdigest(), 'missing_cells': missing})
    return audit


if __name__ == '__main__':
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-json', type=Path, default=here.parents[1] / 'raw_json')
    parser.add_argument('--output', type=Path, default=here / 'output' / 'raw_results')
    args = parser.parse_args()
    result = extract(args.raw_json, args.output)
    print(f"Extracted {result['present']} accuracies; preserved {result['missing']} missing cells.")
