import json
from dataclasses import fields
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src import preprocess as P
from src.config import Config, parse_config
from src.cv import fold_argv, read_folds, stratified_folds
from src.dataset import ImageDataset, PatchDataset, _load_pair
from src.engine import predict_prob
from src.losses import BCEDiceClDiceLoss, SoftClDiceLoss
from src.utils import imwrite


def records_at(tmp_path):
    records = []
    rng = np.random.RandomState(11)
    for i in range(10):
        image = rng.randint(0, 256, (41, 37, 3), dtype=np.uint8)
        mask = np.zeros((41, 37), np.uint8)
        mask[5:35, 3:5+i] = 255
        ip, lp = tmp_path / f'image{i}.png', tmp_path / f'label{i}.png'
        imwrite(str(ip), image)
        imwrite(str(lp), mask)
        records.append(dict(id=str(i), image=str(ip), label=str(lp)))
    return records


def test_seed_changes_membership_and_saved_split_is_validated(tmp_path):
    records = records_at(tmp_path)
    canonical = lambda fs: {frozenset(f) for f in fs}
    first = stratified_folds(records, 5, 42)
    assert canonical(first) != canonical(stratified_folds(records, 5, 123))
    assert canonical(stratified_folds(records, 5, 42, 'legacy')) == canonical(stratified_folds(records, 5, 123, 'legacy'))
    saved = tmp_path / 'split.json'
    saved.write_text(json.dumps({'split': {f'fold{k}': f for k, f in enumerate(first)}}))
    assert read_folds(saved, [r['id'] for r in records]) == first
    with pytest.raises(ValueError):
        read_folds(saved, ['0'])


def test_cv_forwards_all_config_fields_including_empty_and_bool():
    cfg = Config(norm='gn', dropout=0.2, aug_elastic=0.17, vessel_center_prob=0.7,
                 selection_tta=True, thin_center_prob=0.3, init_from='')
    args = SimpleNamespace(**cfg.to_dict())
    actual = parse_config(fold_argv(args, 2, ['3', '9']))
    for field in fields(Config):
        expected = {'seed': cfg.seed+2, 'val_ids': '3,9', 'out_dir': 'outputs/fold2'}.get(field.name, getattr(cfg, field.name))
        if field.name == 'out_dir':
            assert actual.out_dir.replace('\\', '/') == expected
        else:
            assert getattr(actual, field.name) == expected


class Capture(torch.nn.Module):
    def __init__(self):
        super().__init__()
        self.anchor = torch.nn.Parameter(torch.zeros(()))
        self.seen = None

    def forward(self, x):
        self.seen = x.detach().clone()
        return x[:, :1] + self.anchor


@pytest.mark.parametrize('scale', [1., 1.3, 2.])
def test_training_and_inference_receive_identical_preprocessing(tmp_path, scale):
    rec = records_at(tmp_path)[0]
    cfg = Config(scale=scale, amp=False)
    train, _ = _load_pair(cfg, rec)
    train = P.normalize(train, cfg.norm_mean_list, cfg.norm_std_list)
    item = ImageDataset(cfg, [rec['id']], records=[rec])[0]
    model = Capture()
    prob = predict_prob(model, item['image'], cfg, tta=False, rgb=item['rgb'].numpy())
    h, w = train.shape[-2:]
    np.testing.assert_array_equal(train, model.seen[0, :, :h, :w].numpy())
    assert prob.shape == (41, 37)


def test_old_checkpoint_config_keeps_legacy_prediction():
    cfg = Config.from_dict({'scale': 2., 'amp': False})
    assert cfg.preprocess_mode == 'legacy'
    image = torch.randn(3, 41, 37)
    prob = predict_prob(Capture(), image, cfg, tta=False)
    x = torch.nn.functional.interpolate(image[None], scale_factor=2, mode='bilinear', align_corners=False)
    expected = torch.nn.functional.interpolate(x[:, :1].sigmoid(), size=(41, 37), mode='area')[0, 0]
    np.testing.assert_allclose(prob, expected.numpy(), atol=1e-7)


def test_cldice_penalizes_missing_branch_and_has_finite_empty_gradients():
    target = torch.zeros(1, 1, 32, 32)
    target[:, :, 5:27, 15] = 1
    target[:, :, 16, 6:26] = 1
    good = target * 16 - 8
    broken = good.clone()
    broken[:, :, 16, 6:14] = -8
    criterion = SoftClDiceLoss(5)
    assert criterion(good, target) < criterion(broken, target)
    logit = torch.randn(2, 1, 24, 24, requires_grad=True)
    loss = BCEDiceClDiceLoss()(logit, torch.zeros_like(logit))
    loss.backward()
    assert torch.isfinite(loss) and torch.isfinite(logit.grad).all()
    assert logit.grad.abs().sum() > 0


def test_thin_sampler_only_uses_training_records(tmp_path):
    records = records_at(tmp_path)
    cfg = Config(thin_center_prob=0.3, patch_size=32, patches_per_epoch=8)
    ds = PatchDataset(cfg, records[:2])
    assert set(ds.thin_centers) == {'0', '1'}
    assert all(len(x) for x in ds.thin_centers.values())
    x, y = ds[0]
    assert x.shape == (3, 32, 32) and y.shape == (1, 32, 32)
    assert torch.isfinite(x).all() and set(y.unique().tolist()) <= {0., 1.}


def test_submission_transform_matches_official_encoder(tmp_path):
    from PIL import Image
    from segmentation_to_csv import turn_to_str
    from src.evaluation import official_binarize
    from src.rle import rle_encode, validate_rle
    mask = np.zeros((584, 565), np.uint8)
    mask[18:540, 117] = 255
    mask[250:253, 25:520] = 255
    path = tmp_path/'vessels.png'
    Image.fromarray(mask).save(path)
    encoded = rle_encode(official_binarize(mask))
    assert turn_to_str([str(path)]) == [encoded]
    assert validate_rle(encoded)['ok']


def test_oof_cache_covers_only_explicit_folds_and_rejects_leakage(tmp_path, monkeypatch):
    from scripts import round7_audit as audit
    records = records_at(tmp_path)
    split = {f'fold{k}': [str(2*k), str(2*k+1)] for k in range(5)}
    run = tmp_path/'run'
    run.mkdir()
    (run/'cv_summary.json').write_text(json.dumps({'split': split, 'evaluated_folds': [0, 2]}))
    for k in [0, 2]:
        (run/f'fold{k}').mkdir()
        (run/f'fold{k}'/'best.pt').write_bytes(b'fingerprint fixture')
    calls = []
    def load(path, device):
        k = int(path.parent.name.removeprefix('fold'))
        calls.append(k)
        cfg = Config(amp=False, val_ids=','.join(split[f'fold{k}']))
        return Capture(), cfg, {'train_ids': [r['id'] for r in records if r['id'] not in split[f'fold{k}']]}
    monkeypatch.setattr(audit, 'load_checkpoint', load)
    probs = audit.cache_oof(run, 'checkpoint', tmp_path/'cache', records, torch.device('cpu'))
    assert set(probs) == {'0','1','4','5'} and calls == [0, 2]
    audit.cache_oof(run, 'checkpoint', tmp_path/'cache', records, torch.device('cpu'))
    assert calls == [0, 2]
    (run/'fold0'/'best.pt').write_bytes(b'changed model')
    def leaky(path, device):
        model, cfg, extra = load(path, device)
        extra['train_ids'].append(cfg.val_ids.split(',')[0])
        return model, cfg, extra
    monkeypatch.setattr(audit, 'load_checkpoint', leaky)
    with pytest.raises(ValueError, match='train/val overlap'):
        audit.cache_oof(run, 'checkpoint', tmp_path/'cache', records, torch.device('cpu'))


def test_training_rejects_duplicate_validation_ids(tmp_path, monkeypatch):
    import src.train as training
    monkeypatch.setattr(training, 'discover_ids', lambda cfg: ['1', '2', '3'])
    with pytest.raises(ValueError, match='重复或未知'):
        training.main(['--device=cpu', f'--out_dir={tmp_path}', '--val_ids=1,1'])


def test_cv_refuses_partial_oof_when_checkpoint_missing(tmp_path):
    from src.cv import oof_evaluate
    with pytest.raises(FileNotFoundError, match='部分折'):
        oof_evaluate(SimpleNamespace(out_dir=str(tmp_path)), [['1']], [], torch.device('cpu'))
