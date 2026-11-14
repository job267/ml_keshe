"""Round7: fingerprinted OOF probability cache and predefined ensemble audit."""

import argparse
import csv
import hashlib
import json
from pathlib import Path
import sys

import cv2
import numpy as np
import torch
from skimage.morphology import skeletonize
from scipy.ndimage import distance_transform_edt

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0, str(ROOT))

from src.config import Config
from src.cv import read_folds
from src.dataset import ImageDataset, build_records
from src.engine import predict_prob
from src.evaluation import gt_bilinear, gt_nearest, official_binarize
from src.metrics import dice_coef, evaluate_binary
from src.models import load_checkpoint
from src.postprocess import postprocess
from src.utils import fov_mask, get_device, imread_gray, imread_rgb


def sha256(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def write_json(path, value):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def write_csv(path, rows):
    path = Path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    with path.open('w', newline='', encoding='utf-8-sig') as f:
        writer = csv.DictWriter(f, fieldnames=list(rows[0]))
        writer.writeheader()
        writer.writerows(rows)


def source_hashes():
    return {p.relative_to(ROOT).as_posix(): sha256(p) for p in sorted((ROOT/'src').rglob('*.py'))}


def cache_oof(run, mode, out_dir, records, device):
    """Cache invalidates on model/data/source/config changes; float32 probabilities."""
    run, out_dir = Path(run), Path(out_dir)
    ids = [r['id'] for r in records]
    folds = read_folds(run/'cv_summary.json', ids)
    summary = json.loads((run/'cv_summary.json').read_text(encoding='utf-8'))
    evaluated = summary.get('evaluated_folds', list(range(len(folds))))
    if not evaluated or len(set(evaluated)) != len(evaluated) or any(k not in range(len(folds)) for k in evaluated):
        raise ValueError('invalid evaluated_folds')
    ids = [rid for k in evaluated for rid in folds[k]]
    checkpoints = {k: run/f'fold{k}'/'best.pt' for k in evaluated}
    fingerprint = {'run': str(run.resolve()), 'mode': mode, 'tta': True,
                   'split': folds, 'evaluated_folds': evaluated,
                   'checkpoints': {str(p): sha256(p) for p in checkpoints.values()},
                   'source': source_hashes(),
                   'data': {r['id']: [sha256(r['image']), sha256(r['label'])] for r in records}}
    key = hashlib.sha256(json.dumps(fingerprint, sort_keys=True).encode()).hexdigest()
    folder = out_dir/'cache'/f'{run.name}_{mode}'
    meta, archive = folder/'metadata.json', folder/'probabilities.npz'
    if meta.exists() and archive.exists() and json.loads(meta.read_text(encoding='utf-8')).get('key') == key:
        with np.load(archive) as cached:
            if set(cached.files) == set(ids):
                print('cache hit', run, mode, flush=True)
                return {rid: cached[rid] for rid in ids}
    probs = {}
    for k in evaluated:
        val_ids = folds[k]
        model, cfg, extra = load_checkpoint(checkpoints[k], device=device)
        if set(filter(None, cfg.val_ids.split(','))) != set(val_ids):
            raise ValueError(f'{checkpoints[k]}: saved val_ids mismatch')
        if 'train_ids' in extra and set(extra['train_ids']) & set(val_ids):
            raise ValueError(f'{checkpoints[k]}: train/val overlap')
        if mode != 'checkpoint':
            cfg.preprocess_mode = mode
        ds = ImageDataset(cfg, val_ids, records=records)
        for i in range(len(ds)):
            item = ds[i]
            probs[item['id']] = predict_prob(model, item['image'], cfg, device=device,
                                             tta=True, rgb=item['rgb'].numpy()).astype(np.float32)
        del model
        torch.cuda.empty_cache()
        print('cached', run.name, mode, f'fold{k}', flush=True)
    if set(probs) != set(ids):
        raise ValueError('OOF coverage incomplete')
    folder.mkdir(parents=True, exist_ok=True)
    np.savez_compressed(archive, **probs)
    write_json(meta, {'key': key, **fingerprint})
    return probs


def ground_truth(records):
    result = {}
    for r in records:
        gt = imread_gray(r['label']) > 127
        radius = cv2.distanceTransform(gt.astype(np.uint8), cv2.DIST_L2, 5)
        skel = skeletonize(gt)
        # 按最近骨架点的半径定义细血管，避免把所有粗血管边缘都当细血管。
        if skel.any():
            _, nearest = distance_transform_edt(~skel, return_indices=True)
            thin = gt & (radius[tuple(nearest)] <= 1.5)
        else:
            thin = np.zeros_like(gt)
        result[r['id']] = {'gt': gt, 'fov': fov_mask(imread_rgb(r['image'])),
                           'bi': gt_bilinear(gt), 'nn': gt_nearest(gt),
                           'skel': skel, 'thin': thin}
    return result


def score_probs(tag, probs, truth, threshold=.45, min_size=10):
    rows = []
    for rid in sorted(probs, key=int):
        t = truth[rid]
        pred = postprocess(probs[rid], t['fov'], threshold, min_size, False)
        official = official_binarize(pred)
        row = {'candidate': tag, 'id': rid, **evaluate_binary(pred, t['gt']),
               'dice_512_proxy': dice_coef(official, t['bi']),
               'dice_512_nearest': dice_coef(official, t['nn']),
               'precision': float((pred & t['gt']).sum() / max(1, pred.sum())),
               'thin_recall': float((pred & t['thin']).sum() / max(1, t['thin'].sum())),
               'skeleton_recall': float((pred & t['skel']).sum() / max(1, t['skel'].sum())),
               'area': float(pred.mean()), 'tp': int((pred & t['gt']).sum()),
               'pred_pixels': int(pred.sum()), 'gt_pixels': int(t['gt'].sum())}
        rows.append(row)
    summary = {'candidate': tag, 'n_images': len(rows)}
    for key in rows[0]:
        if key not in ('candidate', 'id'):
            summary[key] = float(np.mean([r[key] for r in rows]))
    summary['micro_dice'] = 2*sum(r['tp'] for r in rows)/sum(r['pred_pixels']+r['gt_pixels'] for r in rows)
    return rows, summary


def paired(rows_a, rows_b, fold_by_id, metric='dice_512_proxy'):
    a, b = {r['id']: r for r in rows_a}, {r['id']: r for r in rows_b}
    ids = sorted(set(a) & set(b), key=int)
    if len(ids) != len(a) or len(ids) != len(b):
        raise ValueError('paired evaluation requires identical coverage')
    delta = np.array([a[i][metric]-b[i][metric] for i in ids])
    folds = sorted(set(fold_by_id[i] for i in ids))
    fold_deltas = np.array([np.mean([a[i][metric]-b[i][metric] for i in ids if fold_by_id[i] == f]) for f in folds])
    bootstrap = np.random.RandomState(20261101).choice(fold_deltas, (5000, len(folds)), replace=True).mean(1)
    return {'candidate': rows_a[0]['candidate'], 'reference': rows_b[0]['candidate'],
            'metric': metric, 'mean_delta': float(delta.mean()), 'image_wins': int((delta>0).sum()),
            'fold_wins': int((fold_deltas>0).sum()), 'n_folds': len(folds),
            'fold_bootstrap_low': float(np.quantile(bootstrap,.025)),
            'fold_bootstrap_high': float(np.quantile(bootstrap,.975)),
            'interval_scope': f'fixed trained models, {len(folds)} fold clusters; not training-seed uncertainty'}


def main(argv=None):
    parser = argparse.ArgumentParser()
    parser.add_argument('--out_dir', default='experiments/round7/audit')
    parser.add_argument('--device', default='cuda')
    args = parser.parse_args(argv)
    device = get_device(args.device)
    records = build_records('train/image', 'train/label')
    runs = {'unet': 'outputs_cv', 'attention': 'outputs_cv_att', 'ag_scse': 'outputs_cv_full',
            'strong': 'outputs_cv_sa', 'external_big': 'outputs_cv_sa_big', 'scale2_legacy': 'outputs_cv_s2'}
    out_dir = Path(args.out_dir)
    truth = ground_truth(records)
    predictions = {tag: cache_oof(run, 'checkpoint', out_dir, records, device) for tag, run in runs.items()}
    predictions['scale2_rgb_first'] = cache_oof('outputs_cv_s2', 'rgb_first', out_dir, records, device)
    combinations = {'unet_strong': ['unet','strong'], 'unet_scale2_legacy': ['unet','scale2_legacy'],
                    'unet_scale2_rgb_first': ['unet','scale2_rgb_first'],
                    'strong_scale2_rgb_first': ['strong','scale2_rgb_first'],
                    'all6_equal_runs': list(runs)}
    for tag, members in combinations.items():
        predictions[tag] = {rid: np.mean([predictions[m][rid] for m in members], axis=0) for rid in truth}
    details, summaries, by_tag = [], [], {}
    for tag, probs in predictions.items():
        rows, summary = score_probs(tag, probs, truth)
        by_tag[tag] = rows
        details.extend(rows)
        summaries.append(summary)
        print(tag, 'native', round(summary['dice'],6), 'proxy', round(summary['dice_512_proxy'],6), flush=True)
    splits = read_folds('outputs_cv/cv_summary.json', list(truth))
    fold_by_id = {rid: k for k, ids in enumerate(splits) for rid in ids}
    pairs = [paired(rows, by_tag['unet'], fold_by_id) for tag, rows in by_tag.items() if tag != 'unet']
    pairs.append(paired(by_tag['scale2_rgb_first'], by_tag['scale2_legacy'], fold_by_id))
    write_csv(out_dir/'per_image.csv', details)
    write_csv(out_dir/'summary.csv', summaries)
    write_csv(out_dir/'paired.csv', pairs)
    write_json(out_dir/'protocol.json', {'threshold': .45, 'min_size': 10, 'tta': True,
                                       'hidden_gt_bilinear': 'unconfirmed proxy', 'combinations': combinations})


if __name__ == '__main__':
    main()
