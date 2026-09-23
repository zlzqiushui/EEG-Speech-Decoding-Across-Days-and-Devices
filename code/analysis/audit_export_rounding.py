"""Compare full-precision statistics with the historical two-decimal export."""
from pathlib import Path

import pandas as pd

import reanalyze_submission as analysis
from fixed_source import build_fixed_source

FAMILIES = (
    'zero_shot_vs_chance', 'paired_generalization_penalty',
    'paired_sparse_gain_vs_zero_shot', 'paired_s2_vs_s3',
    'fixed_source_comparison',
)
DISPLAY_COLUMNS = ('mean', 'mean_diff', 'ci_low', 'ci_high',
                   'day2_mean', 'day3_mean', 'delta_mean',
                   'target_mean', 'zero_mean')


def audit(output):
    output = Path(output)
    rounded = output / 'statistics' / 'rounded_export'
    rounded.mkdir(parents=True, exist_ok=True)
    original_raw, original_out = analysis.RAW, analysis.OUT
    try:
        analysis.RAW = output / 'raw_results'
        analysis.OUT = rounded
        analysis.build_statistics()
        build_fixed_source(analysis.RAW, rounded)
    finally:
        analysis.RAW, analysis.OUT = original_raw, original_out

    differences, display_changes = {}, []
    for name in FAMILIES:
        full = pd.read_csv(output / 'statistics' / f'{name}.csv')
        exported = pd.read_csv(rounded / f'{name}.csv')
        pd.testing.assert_frame_equal(full.select_dtypes(exclude='number'),
                                      exported.select_dtypes(exclude='number'))
        columns = full.select_dtypes(include='number').columns
        differences[name] = max(float((full[c] - exported[c]).abs().max()) for c in columns)
        if not (full['p_holm'] < .05).equals(exported['p_holm'] < .05):
            raise AssertionError(f'Holm decision changed: {name}')
        for column in DISPLAY_COLUMNS:
            if column in full:
                for i in range(len(full)):
                    exact, old = round(float(full.loc[i, column]), 1), round(float(exported.loc[i, column]), 1)
                    if exact != old:
                        display_changes.append({'family': name, 'row': i,
                                                'feature': full.loc[i, 'feature'],
                                                'column': column, 'full': exact,
                                                'rounded_export': old})
    return {'max_abs_numeric_difference': max(differences.values()),
            'families': differences, 'holm_decisions_unchanged': True,
            'one_decimal_display_changes': display_changes}
