"""Fixed-schedule, single-variable Round7 ablations; resumable at completed folds.

Run from segmentation/: python -u scripts/round7_experiments.py
Each arm has identical data/schedule; best.pt is explicitly the final epoch.
"""

import argparse
import contextlib
import dataclasses
import gc
import json
from pathlib import Path
import sys
import time

import numpy as np
import torch

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.cv import read_folds
from src.dataset import build_records
from src.train import main as train_main
from scripts.round7_audit import (cache_oof, ground_truth, paired, score_probs,
                                  sha256, source_hashes, write_csv, write_json)


VARIANTS = {'baseline': {}, 'cldice': {'loss': 'bce_dice_cldice'},
            'thin': {'thin_center_prob': .3}}


class Tee:
    def __init__(self, console, log):
        self.console, self.log = console, log

    def write(self, data):
        self.console.write(data)
        self.log.write(data)
        self.log.flush()

    def flush(self):
        self.console.flush()
        self.log.flush()


def train_arm(tag, changes, args, records, folds, protocol_key):
    run = Path(args.out_dir)/f'seed{args.seed}'/tag
    run.mkdir(parents=True, exist_ok=True)
    base = Config(epochs=args.epochs, lr=5e-4, strong_aug=True, cache_dtype='float16',
                  init_from='outputs/pretrained.pt', seed=args.seed, split_strategy='saved',
                  val_every=20, selection_tta=True, selection_metric='dice_512_proxy',
                  checkpoint_policy='last', device='cuda', **changes)
    for fold, val_ids in enumerate(folds):
        if fold not in args.fold_indices:
            continue
        folder = run/f'fold{fold}'
        cfg = dataclasses.replace(base, out_dir=str(folder), seed=args.seed+fold,
                                  val_ids=','.join(val_ids))
        completed = folder/'completed.json'
        if completed.exists():
            done = json.loads(completed.read_text(encoding='utf-8'))
            if (done.get('protocol_key') != protocol_key or done.get('cfg') != cfg.to_dict()
                    or done.get('checkpoint_sha256') != sha256(folder/'best.pt')):
                raise ValueError(f'{folder}: completed-fold provenance mismatch')
            print('resume completed', folder, flush=True)
            continue
        if (folder/'best.pt').exists() or (folder/'training_metadata.json').exists():
            raise RuntimeError(f'{folder}: incomplete existing fold; retain it and use another out_dir')
        folder.mkdir(parents=True, exist_ok=True)
        argv = [f'--{key}={value}' for key, value in cfg.to_dict().items()]
        started = time.monotonic()
        print(f'BEGIN {tag} seed={args.seed} fold={fold} final_epoch={args.epochs}', flush=True)
        with (folder/'console.log').open('w', encoding='utf-8') as log:
            with contextlib.redirect_stdout(Tee(sys.stdout, log)):
                train_main(argv)
        write_json(completed, {'protocol_key': protocol_key, 'cfg': cfg.to_dict(),
                               'checkpoint_sha256': sha256(folder/'best.pt'),
                               'seconds': time.monotonic()-started})
        gc.collect()
        torch.cuda.empty_cache()
    write_json(run/'cv_summary.json', {'split': {f'fold{k}': ids for k, ids in enumerate(folds)},
                                       'config': base.to_dict(), 'protocol_key': protocol_key,
                                       'evaluated_folds': args.fold_indices,
                                       'checkpoint_policy': 'last'})
    return run


def evaluate_arms(runs, args, records, folds):
    dest = Path(args.out_dir)/f'seed{args.seed}'/'evaluation'
    truth = ground_truth(records)
    by_tag, summaries, details = {}, [], []
    for tag, run in runs.items():
        probs = cache_oof(run, 'checkpoint', dest, records, torch.device('cuda'))
        rows, summary = score_probs(tag, probs, truth)
        by_tag[tag] = rows
        details.extend(rows)
        summaries.append(summary)
        print('RESULT', tag, json.dumps(summary), flush=True)
    fold_by_id = {rid: k for k, ids in enumerate(folds) for rid in ids}
    comparisons = [paired(rows, by_tag['baseline'], fold_by_id, metric)
                   for tag, rows in by_tag.items() if tag != 'baseline'
                   for metric in ['dice_512_proxy', 'dice', 'thin_recall']]
    write_csv(dest/'per_image.csv', details)
    write_csv(dest/'summary.csv', summaries)
    if comparisons:
        write_csv(dest/'paired.csv', comparisons)
    return summaries


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='experiments/round7/ablations')
    parser.add_argument('--seed', type=int, default=42)
    parser.add_argument('--epochs', type=int, default=60)
    parser.add_argument('--variants', default='baseline,cldice,thin')
    parser.add_argument('--split_file', default='outputs_cv/cv_summary.json')
    parser.add_argument('--fold_indices', default='0,1,2,3,4', help='初筛可设 0,2；报告明确记录部分覆盖')
    args = parser.parse_args(argv)
    variants = args.variants.split(',')
    if 'baseline' not in variants or len(set(variants)) != len(variants) or any(v not in VARIANTS for v in variants):
        raise ValueError('unique variants must include baseline and exist in VARIANTS')
    records = build_records('train/image', 'train/label')
    folds = read_folds(args.split_file, [r['id'] for r in records])
    args.fold_indices = [int(k) for k in args.fold_indices.split(',')]
    if len(set(args.fold_indices)) != len(args.fold_indices) or not args.fold_indices or any(k not in range(len(folds)) for k in args.fold_indices):
        raise ValueError('invalid fold_indices')
    protocol = {'seed': args.seed, 'epochs': args.epochs, 'variants': {v: VARIANTS[v] for v in variants},
                'source': source_hashes(), 'runner_sha256': sha256(__file__),
                'pretrain_sha256': sha256('outputs/pretrained.pt'),
                'split': folds, 'evaluated_folds': args.fold_indices,
                'checkpoint_policy': 'last', 'postprocess': {'tta': True, 'threshold': .45, 'min_size': 10},
                'data': {r['id']: [sha256(r['image']), sha256(r['label'])] for r in records},
                'primary_metric': 'dice_512_proxy (hidden GT processing remains unconfirmed)',
                'secondary_metrics': ['dice', 'precision', 'thin_recall', 'skeleton_recall'],
                'selection_note': 'fixed final epoch; no selection on outer-fold scores'}
    import hashlib
    key = hashlib.sha256(json.dumps(protocol, sort_keys=True).encode()).hexdigest()
    protocol_path = Path(args.out_dir)/f'seed{args.seed}'/'protocol.json'
    if protocol_path.exists() and json.loads(protocol_path.read_text(encoding='utf-8')) != protocol:
        raise RuntimeError('Protocol changed since run began. Use a different output directory.')
    write_json(protocol_path, protocol)
    runs = {tag: train_arm(tag, VARIANTS[tag], args, records, folds, key) for tag in variants}
    evaluate_arms(runs, args, records, folds)
    write_json(protocol_path.parent/'completed.json', {'protocol_key': key, 'runs': {k: str(v) for k, v in runs.items()}})


if __name__ == '__main__':
    main()
