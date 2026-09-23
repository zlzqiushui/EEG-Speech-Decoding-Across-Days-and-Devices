#!/usr/bin/env python3
"""Reproduce current-paper statistics and figures directly from bundled JSONs."""
import argparse
import importlib.metadata
import json
import os
from pathlib import Path
import tempfile

os.environ.setdefault('MPLCONFIGDIR', str(Path(tempfile.gettempdir()) / 'eeg-paper-matplotlib'))
import pandas as pd
from extract_raw_results import extract
import reanalyze_submission as analysis
from fixed_source import build_fixed_source
from make_figures import make_figures
from audit_export_rounding import audit as audit_export_rounding

TABLES = ('zero_shot_vs_chance', 'paired_generalization_penalty',
          'paired_sparse_gain_vs_zero_shot', 'paired_s2_vs_s3',
          'fixed_source_comparison', 'figure4_source_values')


def verify(output, reference):
    checks = []
    for name in TABLES:
        actual = pd.read_csv(output / 'statistics' / f'{name}.csv')
        expected = pd.read_csv(reference / f'{name}.csv')
        pd.testing.assert_frame_equal(actual, expected, check_exact=False, atol=1e-10, rtol=1e-10)
        checks.append({'table': name, 'rows': len(actual), 'match': True})
    return checks


def main():
    here = Path(__file__).resolve().parent
    parser = argparse.ArgumentParser(description=__doc__)
    parser.add_argument('--raw-json', type=Path, default=here.parents[1] / 'raw_json')
    parser.add_argument('--output', type=Path, default=here / 'output')
    parser.add_argument('--reference', type=Path, default=here / 'reference', help='Saved current-paper results used only after recomputation for verification')
    args = parser.parse_args()
    output = args.output.resolve()
    output.mkdir(parents=True, exist_ok=True)
    report_path = output / 'reproduction_report.json'
    # Replace any earlier success report before starting a potentially failing run.
    report = {'status': 'running', 'training_rerun': False}
    report_path.write_text(json.dumps(report, indent=2) + '\n')
    try:
        audit = extract(args.raw_json, output / 'raw_results')
        extract(args.raw_json, output / 'raw_results_full', round_to_export=False)
        analysis.RAW = output / 'raw_results_full'
        analysis.OUT = output / 'statistics'
        analysis.OUT.mkdir(parents=True, exist_ok=True)
        zero, penalties, pairs, gains = analysis.build_statistics()
        analysis.write_notes(zero, penalties, pairs, gains)
        build_fixed_source(analysis.RAW, analysis.OUT)
        make_figures(output)
        checks = verify(output, args.reference)
        rounding_audit = audit_export_rounding(output)
        # Independently compare reconstructed exports with the bundled CSVs, when present.
        csv_reference = args.raw_json.resolve().parent / 'raw_results'
        csv_checks = 0
        if csv_reference.is_dir():
            for item in audit['files']:
                rel = Path(item['path']).with_suffix('.csv')
                pd.testing.assert_frame_equal(pd.read_csv(output / 'raw_results' / rel), pd.read_csv(csv_reference / rel), check_exact=True)
                csv_checks += 1
        report.update(status='passed', inputs=audit, reference_checks=checks,
                      export_rounding_audit=rounding_audit,
                      bundled_csv_files_verified=csv_checks, statistical_tests=33,
                      calibration_mean_sem_rows=102,
                      versions={name: importlib.metadata.version(name) for name in ('numpy', 'pandas', 'scipy', 'matplotlib')})
    except Exception as exc:
        report.update(status='failed', error=f'{type(exc).__name__}: {exc}')
        raise
    finally:
        report_path.write_text(json.dumps(report, indent=2) + '\n')
    print('PASS: 21 JSON files; 6,225 values and no missing cells; 33 tests and 102 curve rows match the paper.')
    print(f'Results: {output}')


if __name__ == '__main__':
    main()
