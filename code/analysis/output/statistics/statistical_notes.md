# Statistical re-analysis notes

- Inferential unit: participant (N=25). The two cross-day directions are averaged within participant before testing.
- Error bars: SEM for accuracy estimates; 95% t confidence intervals for paired differences.
- Multiplicity: Holm correction is applied separately to zero-shot-vs-chance (6 tests), S2-vs-S3 (12 tests), sparse S2-vs-S1 (6 tests), and generalization penalties (6 tests).
- Data ratios are deterministic, nested, chronological prefixes of target training segments. They are not random trial samples.
- S5 is reported only at r=1.0 because its dataset merge bypasses the ratio-filtered index list.
- All analyses use mel_10; the discontinued 80-band Mel results are excluded.

## Key paired results

- Envelope, cross-day: loss 17.9 pp (95% CI 15.2 to 20.6), n=25, Holm p=6.01e-12, dz=2.71.
- Envelope, day-plus-device: loss 16.3 pp (95% CI 10.5 to 22.1), n=25, Holm p=5.41e-06, dz=1.16.
- Mel-10, cross-day: loss 18.1 pp (95% CI 15.2 to 20.9), n=25, Holm p=9.64e-12, dz=2.62.
- Mel-10, day-plus-device: loss 16.8 pp (95% CI 11.2 to 22.5), n=25, Holm p=4.6e-06, dz=1.23.
- wav2vec 2.0, cross-day: loss 12.2 pp (95% CI 9.0 to 15.3), n=25, Holm p=1.41e-07, dz=1.59.
- wav2vec 2.0, day-plus-device: loss 15.9 pp (95% CI 11.2 to 20.6), n=25, Holm p=9.57e-07, dz=1.40.

## Sparse-calibration gain (S2 - S1 at r=0.2)

- Envelope, cross-day: +10.8 pp (95% CI 7.7 to 13.9), Holm p=8.75e-07.
- Envelope, day-plus-device: +9.9 pp (95% CI 4.8 to 15.0), Holm p=0.00155.
- Mel-10, cross-day: +11.3 pp (95% CI 8.3 to 14.4), Holm p=3.81e-07.
- Mel-10, day-plus-device: +11.4 pp (95% CI 6.3 to 16.4), Holm p=0.000426.
- wav2vec 2.0, cross-day: +3.4 pp (95% CI 0.6 to 6.3), Holm p=0.0198.
- wav2vec 2.0, day-plus-device: +4.9 pp (95% CI 1.4 to 8.4), Holm p=0.0152.
