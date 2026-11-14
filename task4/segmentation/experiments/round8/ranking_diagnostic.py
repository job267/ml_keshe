"""Exploratory native-resolution AP; no parameter search or promotion changes."""
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = Path('experiments/round8')
sys.path.insert(0, str(ROOT/OUT/'implementation'))

import numpy as np
import sklearn
from sklearn.metrics import average_precision_score
from src.dataset import build_records
from src.utils import imread_gray, imread_rgb, fov_mask
from scripts.round7_audit import paired, write_csv, write_json


def main():
    if json.loads((OUT/'status.json').read_text(encoding='utf-8'))['stage'] != 'complete':
        raise RuntimeError('Wait for the staged experiment to finish')
    split = json.loads((OUT/'split.json').read_text(encoding='utf-8'))['split']
    ids = sorted(split['fold0']+split['fold1'], key=int)
    fold_by_id = {rid: int(name.removeprefix('fold')) for name, values in split.items() for rid in values}
    records = {r['id']: r for r in build_records('train/image', 'train/label')}
    truth = {rid: (imread_gray(records[rid]['label']) > 127,
                   fov_mask(imread_rgb(records[rid]['image']))) for rid in ids}
    details, summaries, by_tag = [], [], {}
    for tag in ['baseline', 'fr_unet', 'auxiliary', 'dsconv']:
        folder = OUT/'probability_cache/cache'/f'{tag}_checkpoint'
        metadata = json.loads((folder/'metadata.json').read_text(encoding='utf-8'))
        if Path(metadata['run']).resolve() != (OUT/'seed42'/tag).resolve():
            raise ValueError('Expected seed42 cache')
        with np.load(folder/'probabilities.npz') as cache:
            if set(cache.files) != set(ids):
                raise ValueError('Expected the original eight screening images')
            rows = []
            for rid in ids:
                gt, fov = truth[rid]
                y, p = gt[fov], cache[rid][fov]
                rows.append({'candidate': tag, 'id': rid,
                             'native_fov_average_precision': float(average_precision_score(y, p)),
                             'native_fov_brier': float(np.mean((p-y.astype(np.float32))**2))})
        details.extend(rows)
        by_tag[tag] = rows
        summary = {'candidate': tag, 'n_images': len(rows),
                   **{metric: float(np.mean([r[metric] for r in rows]))
                      for metric in ['native_fov_average_precision', 'native_fov_brier']}}
        summaries.append(summary)
        print(json.dumps(summary), flush=True)
    comparisons = [paired(rows, by_tag['baseline'], fold_by_id, 'native_fov_average_precision')
                   for tag, rows in by_tag.items() if tag != 'baseline']
    dest = OUT/'evaluation_ranking'
    write_csv(dest/'per_image.csv', details)
    write_csv(dest/'summary.csv', summaries)
    write_csv(dest/'paired.csv', comparisons)
    write_json(dest/'protocol.json', {'scope': 'Exploratory diagnosis after screening; never used for promotion',
                                    'resolution': 'native', 'region': 'inside image-derived FOV',
                                    'mask_threshold': 127, 'probabilities': 'raw four-way TTA probabilities',
                                    'min_component_filter': False, 'binary_threshold_search': False,
                                    'AP_definition': 'sklearn non-interpolated average_precision_score',
                                    'sklearn_version': sklearn.__version__, 'exports_submission': False})


if __name__ == '__main__':
    main()
