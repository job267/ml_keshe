"""Fixed-weight OOF diagnostics; never changes promotion decisions or exports CSV."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
OUT = Path('experiments/round8')
os.chdir(ROOT)
sys.path.insert(0, str(ROOT / OUT / 'implementation'))

import torch
from src.dataset import build_records
from scripts.round7_audit import (cache_oof, ground_truth, paired, score_probs,
                                  write_csv, write_json)


def main():
    if json.loads((OUT/'status.json').read_text(encoding='utf-8'))['stage'] != 'complete':
        raise RuntimeError('Wait for the staged training and evaluation to finish')
    spec = json.loads((OUT/'diagnostic_addendum.json').read_text(encoding='utf-8'))
    records = build_records('train/image', 'train/label')
    truth = ground_truth(records)
    split = json.loads((OUT/'split.json').read_text(encoding='utf-8'))['split']
    screen_ids = set(split['fold0'] + split['fold1'])
    tags = ['baseline', 'fr_unet', 'auxiliary', 'dsconv']
    device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')
    probs = {}
    for tag in tags:
        all_probs = cache_oof(OUT/'seed42'/tag, 'checkpoint', OUT/'probability_cache', records, device)
        probs[tag] = {rid: p for rid, p in all_probs.items() if rid in screen_ids}
    coverage = set(probs['baseline'])
    if coverage != screen_ids or any(set(p) != coverage for p in probs.values()):
        raise ValueError('OOF coverage differs across candidates')
    fold_by_id = {rid: int(name.removeprefix('fold')) for name, ids in split.items() for rid in ids}
    details, summaries, comparisons = [], [], []
    baseline_rows, baseline_summary = score_probs('baseline', probs['baseline'], truth)
    details.extend(baseline_rows)
    summaries.append(baseline_summary)
    for tag, weights in spec['mixtures'].items():
        mixed = {rid: sum(weight * probs[name][rid] for name, weight in weights.items())
                 for rid in coverage}
        rows, summary = score_probs(tag, mixed, truth, spec['threshold'], spec['min_size'])
        details.extend(rows)
        summaries.append(summary)
        comparisons.extend(paired(rows, baseline_rows, fold_by_id, metric)
                           for metric in ['dice_512_proxy', 'dice', 'precision', 'thin_recall'])
        print(tag, json.dumps(summary), flush=True)
    dest = OUT/'evaluation_complementarity'
    write_csv(dest/'per_image.csv', details)
    write_csv(dest/'summary.csv', summaries)
    write_csv(dest/'paired.csv', comparisons)
    write_json(dest/'note.json', {'protocol': spec, 'validation_ids': sorted(coverage, key=int),
                                'promotion_gates_changed': False, 'exports_submission': False})


if __name__ == '__main__':
    main()
