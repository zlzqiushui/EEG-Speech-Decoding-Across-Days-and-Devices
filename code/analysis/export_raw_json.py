#!/usr/bin/env python3
"""
export_raw_json.py — export the source result JSONs used by the statistics
(paper_analysis)

The paper's numbers come from only two places:

  Module A (intra-domain DG) : <project>/result_v4/models/<run>/test_results.json
  Module B (SDA)             : <project>/finetune_result_v3/models/<run>/results.json

This script reads the original cluster runs for the paper
(features envelope / mel_10 / wav2vecbase9, seed 42) and merges them into one
human-readable JSON per (module x feature x condition) under `raw_json/`.
No model weights, checkpoints or logs are touched.

Numbers are top-1 match accuracy (1 true vs 4 random distractors, chance = 20%).
`data` holds percentages (2 dp, identical to raw_results/*.csv);
`data_raw` holds the `test_accuracy` fraction. When re-exporting, completed
values in the existing JSON are retained if their individual run files are
not present in the source directory.

Read-only w.r.t. the original result directories.
"""
import os
import re
import json
from collections import defaultdict

SEED = "42"
FEATURES = ["envelope", "mel_10", "wav2vecbase9"]
RATIOS = [0.2, 0.4, 0.6, 0.8, 1.0]
DOMAINS = ["day1", "day2", "brk", "neurascan"]
TRANSFERS = ["day1_to_day2", "day2_to_day1", "neurascan_to_brk"]
STRATS = [1, 2, 3, 4, 5]
N_SUBJECTS = 25

HERE = os.path.dirname(os.path.abspath(__file__))
REPO = os.path.abspath(os.path.join(HERE, "..", ".."))


def _find_project_root(start):
    """Walk up until a directory containing result_v4/models + finetune_result_v3."""
    d = os.path.abspath(start)
    while True:
        if (os.path.isdir(os.path.join(d, "result_v4", "models"))
                and os.path.isdir(os.path.join(d, "finetune_result_v3"))):
            return d
        p = os.path.dirname(d)
        if p == d:
            return None
        d = p


PROJECT = _find_project_root(HERE) or "/gpfs/share/home/2301111611/EEG-Stimulus-Match-Mismatch"
MODA = os.path.join(PROJECT, "result_v4", "models")
MODB = os.path.join(PROJECT, "finetune_result_v3", "models")
OUT = os.path.join(REPO, "raw_json")

patA = re.compile(r'valid\d+-(\w+)-sub(\d+)-ratio([\d.]+)(?:-(brk|neurascan|day1|day2))?-seed(\d+)$')
patB = re.compile(r'valid\d+-(\w+)-sub(\d+)-ratio([\d.]+)-finetune-(\w+)-S(\d)-seed(\d+)$')

STRAT_NAME = {}


def collect_a():
    """d[feature][domain][subject][ratio] = raw test_accuracy (fraction)"""
    d = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for dn in sorted(os.listdir(MODA)):
        m = patA.search(dn)
        if not m or m.group(5) != SEED:
            continue
        f, sub, ratio, dom = m.group(1), int(m.group(2)), float(m.group(3)), m.group(4)
        if f not in FEATURES or dom is None:
            continue
        p = os.path.join(MODA, dn, "test_results.json")
        if not os.path.exists(p):
            continue
        acc = json.load(open(p)).get("test_accuracy")
        if acc is None:
            continue
        d[f][dom][sub][ratio] = acc
    return d


def collect_b():
    """d[feature][transfer][subject][(strategy, ratio)] = raw test_accuracy"""
    d = defaultdict(lambda: defaultdict(lambda: defaultdict(dict)))
    for dn in sorted(os.listdir(MODB)):
        m = patB.search(dn)
        if not m or m.group(6) != SEED:
            continue
        f, sub, ratio, tr, st = m.group(1), int(m.group(2)), float(m.group(3)), m.group(4), int(m.group(5))
        if f not in FEATURES:
            continue
        if st == 5 and ratio != 1.0:
            continue
        p = os.path.join(MODB, dn, "results.json")
        if not os.path.exists(p):
            continue
        j = json.load(open(p))
        acc = j.get("test_accuracy")
        if acc is None:
            continue
        STRAT_NAME[st] = j.get("strategy_name", f"S{st}")
        d[f][tr][sub][(st, ratio)] = acc
    return d


def rk(r):
    return f"r{r:.1f}"


def dump(path, payload):
    os.makedirs(os.path.dirname(path), exist_ok=True)
    if os.path.exists(path):
        with open(path) as fp:
            previous = json.load(fp)
        for subject, columns in previous["data_raw"].items():
            for column, raw_value in columns.items():
                if payload["module"] == "B_sda" and column.startswith("S5_") and column != "S5_r1.0":
                    continue
                if raw_value is None:
                    continue
                current = payload["data_raw"].get(subject, {}).get(column)
                if current is None:
                    payload["data_raw"].setdefault(subject, {})[column] = raw_value
                    payload["data"].setdefault(subject, {})[column] = previous["data"][subject][column]
        payload["n_subjects"] = len(payload["data"])
    with open(path, "w") as fp:
        json.dump(payload, fp, indent=1)
    n = sum(1 for v in payload["data"].values() for x in v.values() if x is not None)
    print(f"  {os.path.relpath(path, OUT):52s} subjects={payload['n_subjects']:2d} values={n}")


COMMON = {
    "seed": 42,
    "metric": "top-1 match accuracy (1 true stimulus vs 4 random distractors, i.e. 5-candidate)",
    "chance_pct": 20.0,
    "value_unit": "percent",
    "note": "data_raw = original 'test_accuracy' field (fraction); data = data_raw * 100, rounded to 2 dp",
}


def main():
    a = collect_a()
    b = collect_b()

    print("=== Module A: intra-domain ===")
    for f in FEATURES:
        for dom in DOMAINS:
            pct, raw = {}, {}
            for sub in range(1, N_SUBJECTS + 1):
                got = a[f].get(dom, {}).get(sub, {})
                p, q = {}, {}
                for r in RATIOS:
                    v = got.get(r)
                    p[rk(r)] = None if v is None else round(v * 100, 2)
                    q[rk(r)] = v
                if all(x is None for x in p.values()):
                    continue
                pct[f"sub-{sub:02d}"] = p
                raw[f"sub-{sub:02d}"] = q
            dump(os.path.join(OUT, "module_A_intra_domain", f, f"{dom}.json"),
                 dict(COMMON, module="A_intra_domain", feature=f, condition=dom,
                      n_subjects=len(pct),
                      source_root="result_v4/models",
                      source_name_template=("BrainNetworkCL-bs32-sl5-ks3-dor0.5-att256-nn32-valid1-"
                                             f"{f}-sub{{N}}-ratio{{R}}-{dom}-seed42/test_results.json"),
                      data=pct, data_raw=raw))

    print("=== Module B: SDA ===")
    for f in FEATURES:
        for tr in TRANSFERS:
            pct, raw = {}, {}
            for sub in range(1, N_SUBJECTS + 1):
                p, q = {}, {}
                for s in STRATS:
                    for r in ([1.0] if s == 5 else RATIOS):
                        v = b[f].get(tr, {}).get(sub, {}).get((s, r))
                        key = f"S{s}_{rk(r)}"
                        p[key] = None if v is None else round(v * 100, 2)
                        q[key] = v
                if all(x is None for x in p.values()):
                    continue
                pct[f"sub-{sub:02d}"] = p
                raw[f"sub-{sub:02d}"] = q
            dump(os.path.join(OUT, "module_B_sda", f, f"{tr}.json"),
                 dict(COMMON, module="B_sda", feature=f, condition=tr,
                      n_subjects=len(pct),
                      strategies={f"S{s}": STRAT_NAME.get(s, f"S{s}") for s in STRATS},
                      source_root="finetune_result_v3/models",
                      source_name_template=("BrainNetworkCL-bs32-sl5-ks3-dor0.5-att256-nn32-valid1-"
                                             f"{f}-sub{{N}}-ratio{{R}}-finetune-{tr}-S{{S}}-seed42/results.json"),
                      data=pct, data_raw=raw))

    print(f"\nDone. Output: {OUT}")


if __name__ == "__main__":
    main()
