"""Audit finished Round8 artifacts against the saved source/data/protection hashes."""
import hashlib
import json
import math
import os
from pathlib import Path
import sys

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = Path('experiments/round8')
sys.path.insert(0, str(ROOT / OUT / 'implementation'))
from src.dataset import build_records, build_external_records


def sha(path):
    return hashlib.sha256(Path(path).read_bytes()).hexdigest()


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


def write(path, value):
    Path(path).write_text(json.dumps(value, ensure_ascii=False, indent=2), encoding='utf-8')


def main():
    assert read(OUT/'status.json')['stage'] == 'complete'
    plan, before = read(OUT/'plan.json'), read(OUT/'before_manifest.json')
    protected = [r for r in before if Path(r['path']).suffix in ('.pt', '.csv')]
    protected_rows = [{**r, 'current_sha256': sha(r['path'])} for r in protected]
    changed_protected = [r['path'] for r in protected_rows if r['sha256'] != r['current_sha256']]
    source_mismatch = [rel for rel, expected in plan['source'].items()
                       if sha(rel) != expected or sha(OUT/'implementation'/rel) != expected]
    runner = Path('scripts/round8_experiments.py')
    runner_ok = sha(runner) == sha(OUT/'implementation'/runner) == plan['runner_sha256']
    frozen_python_mismatch = [p.relative_to(OUT/'implementation').as_posix()
                              for p in (OUT/'implementation').rglob('*.py')
                              if sha(p) != sha(p.relative_to(OUT/'implementation'))]
    target = build_records('train/image', 'train/label')
    external = build_external_records(['data_external/stare', 'data_external/chase', 'data_external/hrf'])
    data_mismatch = []
    for records, expected in [(target, plan['data']), (external, plan['external_data'])]:
        actual = {r['id']: [sha(r['image']), sha(r['label'])] for r in records}
        if actual != expected:
            data_mismatch.extend(rid for rid in actual.keys() | expected.keys()
                                 if actual.get(rid) != expected.get(rid))
    folds = []
    for marker in sorted(OUT.glob('seed*/*/fold*/completed.json')):
        completed = read(marker)
        history = read(marker.parent/'history.json')
        ok = (len(history) == completed['cfg']['epochs']
              and history[-1]['epoch'] == completed['cfg']['epochs']
              and sha(marker.parent/'best.pt') == completed['sha256']
              and all(math.isfinite(r['loss']) for r in history)
              and all(math.isfinite(v) for r in history for k, v in r.items() if k.startswith('val_')))
        folds.append({'path': marker.parent.as_posix(), 'epochs': len(history),
                      'verified': ok, 'seconds_this_session': completed['seconds_this_session']})
    pretrain = read(OUT/'pretrain/fr_unet/completed.json')
    pretrain_history = read(OUT/'pretrain/fr_unet/history.partial.json')
    pretrain_ok = (sha(OUT/'pretrain/fr_unet/pretrained.pt') == pretrain['sha256']
                   and len(pretrain_history) == pretrain['cfg']['epochs']
                   and pretrain_history[-1]['epoch'] == pretrain['cfg']['epochs']
                   and all(math.isfinite(r['loss']) for r in pretrain_history))
    decision = read(OUT/'decision.json')
    candidate_ok = all(Path(path).is_file() for path in decision['candidate_files'])
    changed = [{'path': r['path'], 'before_sha256': r['sha256'], 'after_sha256': sha(r['path'])}
               for r in before if sha(r['path']) != r['sha256']]
    known = {r['path'] for r in before}
    added = []
    for top in ('src', 'scripts', 'tests'):
        for path in Path(top).rglob('*'):
            if path.is_file() and (path.suffix in ('.py', '.json') or path.name.endswith('_LICENSE')):
                if path.as_posix() not in known:
                    added.append({'path': path.as_posix(), 'sha256': sha(path)})
    all_ok = (not changed_protected and not source_mismatch and runner_ok and not frozen_python_mismatch and not data_mismatch
              and len(folds) >= 8 and all(r['verified'] for r in folds) and pretrain_ok and candidate_ok)
    evidence = read(OUT/'verification_before_training.json')
    verification = {'all_checks_passed': all_ok, 'protected_file_count': len(protected_rows),
                    'changed_protected_files': changed_protected, 'protected_files': protected_rows,
                    'source_mismatch': source_mismatch, 'frozen_runner_matches': runner_ok,
                    'frozen_python_mismatch': frozen_python_mismatch,
                    'data_mismatch': data_mismatch, 'target_images': len(target), 'external_images': len(external),
                    'external_target_exact_image_hash_overlap': len({sha(r['image']) for r in target}
                                                                   & {sha(r['image']) for r in external}),
                    'formal_fold_models': folds, 'pretraining_checkpoint_verified': pretrain_ok,
                    'tests_passed_before_training': evidence['tests_passed'],
                    'test_evidence': 'verification_before_training.json',
                    'new_submission_generated': decision['new_submission_generated'],
                    'candidate_files': decision['candidate_files'],
                    'metadata_note': 'model_sources.json descriptions were corrected after the code snapshot; executable source remains unchanged.'}
    write(OUT/'verification.json', verification)
    write(OUT/'change_manifest.json', {'modified_existing_files': changed, 'added_source_files': added})
    print(json.dumps({'all_checks_passed': all_ok, 'protected_files': len(protected_rows),
                      'completed_fold_models': len(folds), 'new_submission_generated': decision['new_submission_generated']}))
    if not all_ok:
        raise RuntimeError('Round8 audit failed; inspect verification.json')


if __name__ == '__main__':
    main()
