#!/usr/bin/env python3
"""Re-analysis for the revised ICASSP submission.

Reads only the exported per-participant CSV files in ``raw_results``.  It
keeps the participant as the inferential unit, averages the two cross-day
directions within participant, reports 95% confidence intervals/effect sizes,
and applies Holm correction within each family of confirmatory comparisons.
No decoder is trained or evaluated by this script.
"""

from pathlib import Path

import numpy as np
import pandas as pd
from scipy import stats


ROOT = Path(__file__).resolve().parents[2]
RAW = Path(__file__).resolve().parent / "output" / "raw_results"
OUT = Path(__file__).resolve().parent / "output" / "statistics"

FEATURES = ["envelope", "mel_10", "wav2vecbase9"]
LABEL = {"envelope": "Envelope", "mel_10": "Mel-10", "wav2vecbase9": "wav2vec 2.0"}
RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]
CHANCE = 20.0


def read_a(feature, domain):
    frame = pd.read_csv(RAW / "module_A_intra_domain" / feature / f"{domain}.csv").set_index("subject")
    if domain != "neurascan" and frame["r1.0"].isna().any():
        raise ValueError(f"Missing reference results: {feature}/{domain}")
    return frame


def read_b(feature, transfer):
    frame = pd.read_csv(RAW / "module_B_sda" / feature / f"{transfer}.csv").set_index("subject")
    if len(frame) != 25 or not frame.index.is_unique or frame.isna().any().any():
        raise ValueError(f"Incomplete participant results: {feature}/{transfer}")
    return frame


def col(strategy, ratio):
    return f"S{strategy}_r{ratio:.1f}"


def mean_sem(x):
    x = pd.Series(x).dropna().astype(float)
    return x.mean(), x.sem(), len(x)


def ci_mean(x, alpha=0.05):
    x = pd.Series(x).dropna().astype(float)
    if len(x) < 2:
        return np.nan, np.nan
    h = stats.t.ppf(1 - alpha / 2, len(x) - 1) * x.sem()
    return x.mean() - h, x.mean() + h


def holm(pvals):
    """Holm adjusted p-values, returned in original order."""
    p = np.asarray(pvals, dtype=float)
    order = np.argsort(p)
    ranked = p[order]
    adjusted = np.maximum.accumulate((len(p) - np.arange(len(p))) * ranked)
    adjusted = np.minimum(adjusted, 1.0)
    out = np.empty_like(adjusted)
    out[order] = adjusted
    return out


def dz(x):
    x = pd.Series(x).dropna().astype(float)
    sd = x.std(ddof=1)
    return x.mean() / sd if sd > 0 else np.nan


def paired_test(a, b):
    z = pd.concat([pd.Series(a), pd.Series(b)], axis=1).dropna()
    d = z.iloc[:, 0] - z.iloc[:, 1]
    t, p = stats.ttest_1samp(d, 0.0)
    lo, hi = ci_mean(d)
    return dict(n=len(d), mean_diff=d.mean(), ci_low=lo, ci_high=hi,
                t=t, p=p, cohen_dz=dz(d))


def build_statistics():
    # Family 1: zero-shot accuracy against 20% chance (six tests).
    zero_rows = []
    for f in FEATURES:
        d12, d21, dev = read_b(f, "day1_to_day2"), read_b(f, "day2_to_day1"), read_b(f, "neurascan_to_brk")
        cross_day = pd.concat([d12[col(1, 1.0)], d21[col(1, 1.0)]], axis=1).mean(axis=1)
        for shift, values in [("cross-day", cross_day), ("day-plus-device", dev[col(1, 1.0)])]:
            values = values.dropna()
            t, p = stats.ttest_1samp(values, CHANCE)
            m, se, n = mean_sem(values)
            lo, hi = ci_mean(values)
            zero_rows.append(dict(feature=LABEL[f], shift=shift, n=n, mean=m, sem=se,
                                  ci_low=lo, ci_high=hi, t=t, p=p))
    padj = holm([r["p"] for r in zero_rows])
    for r, q in zip(zero_rows, padj):
        r["p_holm"] = q
    pd.DataFrame(zero_rows).to_csv(OUT / "zero_shot_vs_chance.csv", index=False)

    # Family 2: paired S2-vs-S3 comparisons at sparse/full calibration.
    pair_rows = []
    gain_rows = []
    for f in FEATURES:
        d12, d21, dev = read_b(f, "day1_to_day2"), read_b(f, "day2_to_day1"), read_b(f, "neurascan_to_brk")
        for shift, frames in [("cross-day", [d12, d21]), ("day-plus-device", [dev])]:
            def participant_values(strategy, ratio):
                return pd.concat([x[col(strategy, ratio)] for x in frames], axis=1).mean(axis=1)
            for ratio in (0.2, 1.0):
                res = paired_test(participant_values(2, ratio), participant_values(3, ratio))
                pair_rows.append(dict(feature=LABEL[f], shift=shift, ratio=ratio, comparison="S2-S3", **res))
            # Sparse calibration gain relative to the ratio-independent zero-shot model.
            res = paired_test(participant_values(2, 0.2), participant_values(1, 1.0))
            gain_rows.append(dict(feature=LABEL[f], shift=shift, ratio=0.2, comparison="S2-S1", **res))
    for rows in (pair_rows, gain_rows):
        padj = holm([r["p"] for r in rows])
        for r, q in zip(rows, padj):
            r["p_holm"] = q
    pd.DataFrame(pair_rows).to_csv(OUT / "paired_s2_vs_s3.csv", index=False)
    pd.DataFrame(gain_rows).to_csv(OUT / "paired_sparse_gain_vs_zero_shot.csv", index=False)

    # Paired target-domain penalty: the target-trained r=1.0 baseline minus S1.
    penalty_rows = []
    for f in FEATURES:
        a1, a2, abrk = read_a(f, "day1"), read_a(f, "day2"), read_a(f, "brk")
        d12, d21, dev = read_b(f, "day1_to_day2"), read_b(f, "day2_to_day1"), read_b(f, "neurascan_to_brk")
        target_cd = pd.concat([a2["r1.0"], a1["r1.0"]], axis=1).mean(axis=1)
        zero_cd = pd.concat([d12[col(1, 1.0)], d21[col(1, 1.0)]], axis=1).mean(axis=1)
        target_dev = abrk["r1.0"]
        zero_dev = dev[col(1, 1.0)]
        for shift, target, zero in [("cross-day", target_cd, zero_cd), ("day-plus-device", target_dev, zero_dev)]:
            res = paired_test(target, zero)
            tm, tse, _ = mean_sem(target)
            zm, zse, _ = mean_sem(zero)
            penalty_rows.append(dict(feature=LABEL[f], shift=shift,
                                     target_mean=tm, target_sem=tse,
                                     zero_mean=zm, zero_sem=zse, **res))
    padj = holm([r["p"] for r in penalty_rows])
    for r, q in zip(penalty_rows, padj):
        r["p_holm"] = q
    pd.DataFrame(penalty_rows).to_csv(OUT / "paired_generalization_penalty.csv", index=False)

    # Paper table: means and SEMs for S1/S2/S3/S5 at r=.2 and r=1.
    table_rows = []
    for f in FEATURES:
        d12, d21, dev = read_b(f, "day1_to_day2"), read_b(f, "day2_to_day1"), read_b(f, "neurascan_to_brk")
        for shift, frames in [("Cross-day", [d12, d21]), ("Day+device", [dev])]:
            row = {"Shift": shift, "Feature": LABEL[f]}
            for s, r in [(1, 1.0), (2, .2), (3, .2), (2, 1.0), (3, 1.0), (5, 1.0)]:
                vals = pd.concat([x[col(s, r)] for x in frames], axis=1).mean(axis=1)
                m, se, n = mean_sem(vals)
                key = f"S{s}_r{r:.1f}"
                row[key] = f"{m:.1f} ({se:.1f})"
                row[key + "_n"] = n
            table_rows.append(row)
    pd.DataFrame(table_rows).to_csv(OUT / "paper_table_sda_mean_sem.csv", index=False)

    return pd.DataFrame(zero_rows), pd.DataFrame(penalty_rows), pd.DataFrame(pair_rows), pd.DataFrame(gain_rows)


def write_notes(zero, penalties, pairs, gains):
    lines = [
        "# Statistical re-analysis notes",
        "",
        "- Inferential unit: participant (N=25). The two cross-day directions are averaged within participant before testing.",
        "- Error bars: SEM for accuracy estimates; 95% t confidence intervals for paired differences.",
        "- Multiplicity: Holm correction is applied separately to zero-shot-vs-chance (6 tests), S2-vs-S3 (12 tests), sparse S2-vs-S1 (6 tests), and generalization penalties (6 tests).",
        "- Data ratios are deterministic, nested, chronological prefixes of target training segments. They are not random trial samples.",
        "- S5 is reported only at r=1.0 because its dataset merge bypasses the ratio-filtered index list.",
        "- All analyses use mel_10; the discontinued 80-band Mel results are excluded.",
        "",
        "## Key paired results",
        "",
    ]
    for _, r in penalties.iterrows():
        lines.append(f"- {r['feature']}, {r['shift']}: loss {r.mean_diff:.1f} pp (95% CI {r.ci_low:.1f} to {r.ci_high:.1f}), n={int(r.n)}, Holm p={r.p_holm:.3g}, dz={r.cohen_dz:.2f}.")
    lines += ["", "## Sparse-calibration gain (S2 - S1 at r=0.2)", ""]
    for _, r in gains.iterrows():
        lines.append(f"- {r['feature']}, {r['shift']}: {r.mean_diff:+.1f} pp (95% CI {r.ci_low:.1f} to {r.ci_high:.1f}), Holm p={r.p_holm:.3g}.")
    (OUT / "statistical_notes.md").write_text("\n".join(lines) + "\n", encoding="utf-8")
