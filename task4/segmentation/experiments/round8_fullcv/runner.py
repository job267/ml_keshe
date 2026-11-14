"""Complete the saved Round8 split for U-Net and auxiliary supervision only."""
import argparse
import dataclasses
from datetime import datetime
import gc
import json
import math
import os
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[2]
os.chdir(ROOT)
OUT = Path('experiments/round8_fullcv')
OLD = Path('experiments/round8')
FROZEN = ROOT/OLD/'implementation'
sys.path.insert(0, str(ROOT))
sys.path.insert(0, str(FROZEN))

import torch
from src.config import Config
from src.cv import read_folds
from src.dataset import build_records
from src.train import main as train_main
from scripts.round8_experiments import digest, run_logged
from scripts.round7_audit import (cache_oof, ground_truth, paired, score_probs,
                                  sha256, write_csv, write_json)

TAGS = ['baseline', 'auxiliary']
METRICS = ['dice_512_proxy', 'dice', 'precision', 'thin_recall', 'skeleton_recall']


def read(path):
    return json.loads(Path(path).read_text(encoding='utf-8'))


class Extension:
    def __init__(self):
        OUT.mkdir(parents=True, exist_ok=True)
        original = read(OLD/'plan.json')
        self.records = build_records('train/image', 'train/label')
        self.ids = {r['id'] for r in self.records}
        self.folds = read_folds(OLD/'split.json', self.ids)
        self.split = {f'fold{k}': values for k, values in enumerate(self.folds)}
        if self.split != original['split']:
            raise ValueError('Original split differs from plan')
        current_data = {r['id']: [sha256(r['image']), sha256(r['label'])] for r in self.records}
        if current_data != original['data']:
            raise ValueError('Target data changed')
        support = {p.relative_to(FROZEN).as_posix(): sha256(p)
                   for top in ['src', 'scripts'] for p in (FROZEN/top).rglob('*.py')}
        for rel, expected in original['source'].items():
            if support.get(rel) != expected:
                raise ValueError(f'Frozen model source changed: {rel}')
        if sha256('outputs/pretrained.pt') != original['baseline_pretrain_sha256']:
            raise ValueError('U-Net initializer changed')
        self.templates = {tag: Config.from_dict(read(OLD/f'seed42/{tag}/fold0/completed.json')['cfg'])
                          for tag in TAGS}
        for tag, cfg in self.templates.items():
            if (cfg.epochs, cfg.scale, cfg.checkpoint_policy, cfg.strong_aug, cfg.seed) != (60, 1., 'last', True, 42):
                raise ValueError(f'Unexpected original recipe: {tag}')
        self.reused = []
        for tag in TAGS:
            for fold in [0, 1]:
                source = OLD/f'seed42/{tag}/fold{fold}'
                marker = read(source/'completed.json')
                expected_signature = digest({'cfg': marker['cfg'], 'initializer': original['baseline_pretrain_sha256'],
                                             'plan': digest(original)})
                if marker['signature'] != expected_signature or sha256(source/'best.pt') != marker['sha256']:
                    raise ValueError(f'Original checkpoint provenance mismatch: {source}')
                self.verify_checkpoint(source, Config.from_dict(marker['cfg']), fold)
                self.reused.append({'tag': tag, 'fold': fold, 'source': source.as_posix(),
                                    'checkpoint_sha256': marker['sha256'],
                                    'completed_sha256': sha256(source/'completed.json')})
        self.configs = {(tag, k): dataclasses.replace(self.templates[tag],
                         out_dir=str(OUT/f'seed42/{tag}/fold{k}'), seed=42+k,
                         val_ids=','.join(self.folds[k]), resume_from='')
                        for tag in TAGS for k in [2, 3, 4]}
        self.plan = {'scope': 'User-authorized completion of baseline and auxiliary folds2/3/4',
                     'runner_sha256': sha256(__file__), 'original_plan_sha256': sha256(OLD/'plan.json'),
                     'frozen_support': support, 'data': current_data, 'split': self.split,
                     'initializer_sha256': original['baseline_pretrain_sha256'],
                     'official_encoder_sha256': sha256('segmentation_to_csv.py'),
                     'training_seeds': [42], 'reused': self.reused,
                     'training_configs': {f'{tag}/fold{k}': cfg.to_dict() for (tag, k), cfg in self.configs.items()},
                     'selection': {'primary': 'dice_512_proxy', 'positive_delta_required_for_trial_csv': True,
                                   'minimum_positive_delta': 0., 'second_seed_required': False,
                                   'native_metrics': 'diagnostic, not a new hard gate'},
                     'hidden_gt_conversion': 'unverified proxy', 'upload_to_kaggle': False}
        if (OUT/'plan.json').exists() and read(OUT/'plan.json') != self.plan:
            raise ValueError('Extension source/data/configuration changed; restore the frozen runner')
        write_json(OUT/'plan.json', self.plan)
        write_json(OUT/'split.json', {'split': self.split})
        frozen_runner = OUT/'runner_frozen.py'
        if not frozen_runner.exists():
            shutil.copy2(__file__, frozen_runner)
        elif sha256(frozen_runner) != self.plan['runner_sha256']:
            raise ValueError('Frozen extension runner differs')
        if not (OUT/'protected_before.json').exists():
            paths = [Path(r['path']) for r in read(OLD/'before_manifest.json')
                     if Path(r['path']).suffix in ['.pt', '.csv']]
            paths += list((OLD/'seed42').glob('*/fold*/best.pt')) + [OLD/'pretrain/fr_unet/pretrained.pt']
            paths += [OLD/name for name in ['plan.json', 'screen_decision.json', 'decision.json', 'status.json']]
            paths += list((OLD/'seed42/evaluation_screen').glob('*.csv'))
            write_json(OUT/'protected_before.json', {p.as_posix(): sha256(p) for p in paths})
        self.check_protected()

    def status(self, stage, **details):
        write_json(OUT/'status.json', {'stage': stage, 'updated_at': datetime.now().astimezone().isoformat(), **details})

    def check_protected(self):
        changed = [p for p, expected in read(OUT/'protected_before.json').items() if sha256(p) != expected]
        if changed:
            raise ValueError(f'Protected originals changed: {changed}')

    def verify_checkpoint(self, folder, cfg, fold):
        raw = torch.load(folder/'best.pt', map_location='cpu', weights_only=False)
        expected = {k: v for k, v in cfg.to_dict().items() if k != 'resume_from'}
        actual = {k: v for k, v in raw['cfg'].items() if k != 'resume_from'}
        history = read(folder/'history.json')
        if (actual != expected or raw['epoch'] != 60 or len(history) != 60
                or any(not math.isfinite(r['loss']) for r in history)
                or any(not math.isfinite(v) for r in history for k, v in r.items() if k.startswith('val_'))):
            raise ValueError(f'Incomplete or different training recipe: {folder}')
        if (set(raw['val_ids']) != set(self.folds[fold]) or set(raw['train_ids']) != self.ids-set(self.folds[fold])
                or len(raw['val_ids']) != 4 or len(raw['train_ids']) != 16):
            raise ValueError(f'Train/validation split mismatch: {folder}')
        del raw

    def prepare(self):
        for item in self.reused:
            source = Path(item['source'])
            dest = OUT/f"seed42/{item['tag']}/fold{item['fold']}"
            dest.mkdir(parents=True, exist_ok=True)
            for name in ['best.pt', 'history.json']:
                if not (dest/name).exists():
                    shutil.copy2(source/name, dest/name)
                elif sha256(dest/name) != sha256(source/name):
                    raise ValueError(f'Reused artifact differs: {dest/name}')
            marker = {'kind': 'reused', 'protocol': digest(self.plan), **item}
            if (dest/'completed.json').exists() and read(dest/'completed.json') != marker:
                raise ValueError('Reused fold marker differs')
            write_json(dest/'completed.json', marker)
        self.summaries()
        write_json(OUT/'preparation_verification.json', {'passed': True, 'reused_checkpoints': 4,
                   'additional_training_folds': 6, 'split_coverage': len(self.ids),
                   'train_validation_partition': '16 train / 4 validation; identical original split',
                   'training_code': str(FROZEN), 'prior_test_evidence': str(OLD/'verification_before_training.json'),
                   'original_minimum_delta_gate_removed': True})
        self.status('prepared', reused_checkpoints=4, additional_training_folds=6)
        print('PREPARED: four verified checkpoints reused; six new folds planned', flush=True)

    def summaries(self):
        for tag in TAGS:
            run = OUT/f'seed42/{tag}'
            done = [k for k in range(5) if (run/f'fold{k}/completed.json').exists()]
            write_json(run/'cv_summary.json', {'split': self.split, 'evaluated_folds': done,
                                              'checkpoint_policy': 'last', 'plan': digest(self.plan)})

    def train_fold(self, tag, fold):
        cfg = self.configs[tag, fold]
        folder = Path(cfg.out_dir)
        signature = digest({'cfg': cfg.to_dict(), 'plan': digest(self.plan)})
        marker = folder/'completed.json'
        if marker.exists():
            done = read(marker)
            if done['signature'] != signature or sha256(folder/'best.pt') != done['checkpoint_sha256']:
                raise ValueError('Completed fold provenance differs')
            self.verify_checkpoint(folder, cfg, fold)
            print('REUSE COMPLETED', tag, fold, flush=True)
            return
        if (folder/'best.pt').exists() and not (folder/'resume.pt').exists():
            raise RuntimeError(f'Incomplete model without resume state: {folder}')
        self.status('training', model=tag, fold=fold, epochs=60,
                    new_models_completed=sum((Path(c.out_dir)/'completed.json').exists() for c in self.configs.values()))
        actual = dataclasses.replace(cfg, resume_from=str(folder/'resume.pt') if (folder/'resume.pt').exists() else '')
        print('BEGIN', tag, f'fold{fold}', 'seed', cfg.seed, flush=True)
        started = time.monotonic()
        run_logged(train_main, actual, folder/'console.log')
        self.verify_checkpoint(folder, cfg, fold)
        write_json(marker, {'kind': 'trained', 'signature': signature, 'cfg': cfg.to_dict(),
                            'checkpoint_sha256': sha256(folder/'best.pt'),
                            'seconds_this_session': time.monotonic()-started})
        self.summaries()

    def evaluate(self):
        self.status('evaluation')
        truth = ground_truth(self.records)
        fold_by_id = {rid: k for k, ids in enumerate(self.folds) for rid in ids}
        probs = {tag: cache_oof(OUT/f'seed42/{tag}', 'checkpoint', OUT/'probability_cache',
                               self.records, torch.device('cuda')) for tag in TAGS}
        if any(set(p) != self.ids for p in probs.values()):
            raise ValueError('Incomplete full OOF coverage')
        scopes = {'full20': self.ids, 'original8': set(self.folds[0]+self.folds[1]),
                  'added12': set(self.folds[2]+self.folds[3]+self.folds[4])}
        comparisons, all_rows = {}, {}
        for label, ids in scopes.items():
            dest = OUT/f'evaluation_{label}'
            rows, summaries, by_tag = [], [], {}
            for tag in TAGS:
                selected = {rid: p for rid, p in probs[tag].items() if rid in ids}
                detail, summary = score_probs(tag, selected, truth)
                rows.extend(detail); summaries.append(summary); by_tag[tag] = detail
                print('RESULT', label, tag, json.dumps(summary), flush=True)
            pairs = [paired(by_tag['auxiliary'], by_tag['baseline'], fold_by_id, metric) for metric in METRICS]
            write_csv(dest/'per_image.csv', rows); write_csv(dest/'summary.csv', summaries)
            write_csv(dest/'paired.csv', pairs)
            comparisons[label] = {p['metric']: p for p in pairs}
            if label == 'full20':
                all_rows = by_tag
        fold_rows = []
        for tag, rows in all_rows.items():
            for k in range(5):
                selected = [r for r in rows if fold_by_id[r['id']] == k]
                fold_rows.append({'candidate': tag, 'fold': k, 'n_images': len(selected),
                                  **{m: sum(r[m] for r in selected)/len(selected) for m in METRICS}})
        write_csv(OUT/'evaluation_full20/per_fold.csv', fold_rows)
        return comparisons

    def export(self, tag, role):
        from scripts.ensemble import main as ensemble_main
        from src.rle import build_submission, validate_rle
        from segmentation_to_csv import turn_to_str
        self.status('test_ensemble', model=tag, role=role)
        dest = OUT/f'{role}_{tag}'
        images = ROOT/dest/'image'
        checkpoints = [str(ROOT/OUT/f'seed42/{tag}/fold{k}/best.pt') for k in range(5)]
        ensemble_main(['--checkpoints', ','.join(checkpoints), '--data_dir', str(ROOT/'test'),
                       '--pred_dir', str(images), '--threshold', '.45', '--min_size', '10', '--device', 'cuda'])
        path = ROOT/dest/f'submission_round8_fullcv_{tag}.csv'
        table = build_submission(str(images), str(path))
        if len(table) != 20 or len(set(table['Id'])) != 20 or not all(validate_rle(s)['ok'] for s in table['Predicted']):
            raise ValueError('Submission validation failed')
        if turn_to_str([str(images/f'{rid}.png') for rid in table['Id']]) != table['Predicted'].tolist():
            raise ValueError('Official encoder differs')
        write_json(dest/'manifest.json', {'role': role, 'model': tag, 'checkpoint_sha256': {p: sha256(p) for p in checkpoints},
                                         'csv_sha256': sha256(path), 'plan': digest(self.plan),
                                         'tta': True, 'threshold': .45, 'min_size': 10, 'uploaded': False})
        gc.collect(); torch.cuda.empty_cache()
        return path.relative_to(ROOT).as_posix()

    def run(self):
        self.prepare()
        for tag in TAGS:
            for fold in [2, 3, 4]:
                self.train_fold(tag, fold)
        comparisons = self.evaluate()
        delta = comparisons['full20']['dice_512_proxy']['mean_delta']
        exports = {}
        if delta > 0:
            exports['candidate'] = self.export('auxiliary', 'candidate')
            exports['baseline_reference'] = self.export('baseline', 'reference')
        self.check_protected()
        for (tag, fold), cfg in self.configs.items():
            marker = read(Path(cfg.out_dir)/'completed.json')
            if sha256(Path(cfg.out_dir)/'best.pt') != marker['checkpoint_sha256']:
                raise ValueError('New checkpoint changed after evaluation')
        write_json(OUT/'decision.json', {'comparisons': comparisons, 'full_proxy_delta': delta,
                   'positive_mean_observed': delta > 0, 'minimum_delta_gate': 0.,
                   'new_submission_generated': bool(exports), 'exports': exports, 'uploaded_to_kaggle': False,
                   'scope': 'One training seed; same 20-image CV, not a new independent test set'})
        write_json(OUT/'verification.json', {'passed': True, 'reused_models': 4, 'newly_trained_models': 6,
                   'folds_per_model': 5, 'oof_images_per_model': 20, 'original_protected_files': len(read(OUT/'protected_before.json')),
                   'originals_unchanged': True, 'data_and_source_plan': digest(self.plan),
                   'official_encoder_matches_plan': sha256('segmentation_to_csv.py') == self.plan['official_encoder_sha256'],
                   'new_submission_generated': bool(exports)})
        self.status('complete', full_proxy_delta=delta, exports=exports)
        print('COMPLETE', json.dumps({'full_proxy_delta': delta, 'exports': exports}), flush=True)


def main():
    parser = argparse.ArgumentParser()
    parser.add_argument('--prepare-only', action='store_true')
    args = parser.parse_args()
    experiment = Extension()
    if args.prepare_only:
        experiment.prepare()
    else:
        experiment.run()


if __name__ == '__main__':
    main()
