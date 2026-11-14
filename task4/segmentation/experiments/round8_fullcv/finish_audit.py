"""Independent artifact/source/protection checks after full-CV completion."""
import csv
import hashlib
import json
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = Path('experiments/round8_fullcv')
OLD = Path('experiments/round8')
FROZEN = ROOT/OLD/'implementation'
sys.path.insert(0, str(FROZEN))
from src.dataset import build_records


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def table(path):
    with Path(path).open(encoding='utf-8-sig', newline='') as handle:
        return list(csv.DictReader(handle))


def main():
    assert read(OUT/'status.json')['stage'] == 'complete'
    plan = read(OUT/'plan.json')
    assert sha(OUT/'runner.py') == sha(OUT/'runner_frozen.py') == plan['runner_sha256']
    support = {p.relative_to(FROZEN).as_posix(): sha(p)
               for top in ['src', 'scripts'] for p in (FROZEN/top).rglob('*.py')}
    assert support == plan['frozen_support']
    assert sha(OLD/'plan.json') == plan['original_plan_sha256']
    assert sha('outputs/pretrained.pt') == plan['initializer_sha256']
    assert sha('segmentation_to_csv.py') == plan['official_encoder_sha256']
    records = build_records('train/image', 'train/label')
    assert {r['id']: [sha(r['image']), sha(r['label'])] for r in records} == plan['data']
    protected = read(OUT/'protected_before.json')
    assert all(sha(path) == expected for path, expected in protected.items())
    checkpoints = []
    for tag in ['baseline', 'auxiliary']:
        assert read(OUT/f'seed42/{tag}/cv_summary.json')['evaluated_folds'] == list(range(5))
        for fold in range(5):
            folder = OUT/f'seed42/{tag}/fold{fold}'
            marker = read(folder/'completed.json')
            assert marker['kind'] == ('reused' if fold < 2 else 'trained')
            assert sha(folder/'best.pt') == marker['checkpoint_sha256']
            assert len(read(folder/'history.json')) == 60
            checkpoints.append({'path': (folder/'best.pt').as_posix(), 'kind': marker['kind'],
                                'sha256': marker['checkpoint_sha256']})
    full = table(OUT/'evaluation_full20/per_image.csv')
    ids = {r['id'] for r in records}
    for tag in ['baseline', 'auxiliary']:
        selected = [r for r in full if r['candidate'] == tag]
        assert len(selected) == 20 and {r['id'] for r in selected} == ids
    old_rows = {(r['candidate'], r['id']): r for r in table(OLD/'seed42/evaluation_screen/per_image.csv')
                if r['candidate'] in ['baseline', 'auxiliary']}
    max_drift = {metric: max(abs(float(r[metric])-float(old_rows[r['candidate'], r['id']][metric]))
                             for r in full if (r['candidate'], r['id']) in old_rows)
                 for metric in ['dice', 'dice_512_proxy', 'precision', 'thin_recall']}
    decision = read(OUT/'decision.json')
    assert decision['new_submission_generated'] == bool(decision['exports'])
    exports = {}
    test_ids = {r['id'] for r in build_records('test/image', 'test/image')}
    for role, path in decision['exports'].items():
        manifest = read(Path(path).parent/'manifest.json')
        rows = table(path)
        assert len(rows) == 20 and {r['Id'] for r in rows} == test_ids
        assert sha(path) == manifest['csv_sha256'] and not manifest['uploaded']
        exports[role] = {'path': path, 'sha256': sha(path)}
    result = {'passed': True, 'protected_original_count': len(protected), 'originals_unchanged': True,
              'frozen_source_and_runner_unchanged': True, 'target_data_unchanged': True,
              'checkpoint_manifest': checkpoints, 'oof_images_per_model': 20,
              'original8_max_absolute_metric_drift': max_drift,
              'new_submission_generated': bool(exports), 'exports': exports}
    (OUT/'finish_audit.json').write_text(json.dumps(result, ensure_ascii=False, indent=2), encoding='utf-8')
    print(json.dumps({key: result[key] for key in ['passed', 'protected_original_count', 'oof_images_per_model',
                                                 'original8_max_absolute_metric_drift', 'new_submission_generated']}))


if __name__ == '__main__':
    main()
