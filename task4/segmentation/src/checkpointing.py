"""Epoch-boundary resume, including optimizer, AMP and patch sampler RNG state."""

import os
import random
import numpy as np
import torch


def save_resume(path, kind, cfg, model, optimizer, scheduler, scaler, dataset, history, **stats):
    state = {'kind': kind, 'cfg': cfg.to_dict(), 'model': model.state_dict(),
             'optimizer': optimizer.state_dict(), 'scheduler': scheduler.state_dict(),
             'scaler': scaler.state_dict(), 'epoch': len(history), 'history': history,
             'stats': stats, 'rng': {'python': random.getstate(), 'numpy': np.random.get_state(),
                                    'torch': torch.get_rng_state(),
                                    'cuda': torch.cuda.get_rng_state_all() if torch.cuda.is_available() else [],
                                    'patches': dataset.rng.get_state()}}
    temporary = path + '.tmp'
    torch.save(state, temporary)
    os.replace(temporary, path)


def load_resume(path, kind, cfg, model, optimizer, scheduler, scaler, dataset, device):
    state = torch.load(path, map_location=device, weights_only=False)
    if state['kind'] != kind:
        raise ValueError('Resume checkpoint kind mismatch')
    ignored = {'resume_from'}
    planned = {k:v for k,v in cfg.to_dict().items() if k not in ignored}
    actual = {k:v for k,v in state['cfg'].items() if k not in ignored}
    if planned != actual:
        raise ValueError('Resume configuration differs from the saved training protocol')
    model.load_state_dict(state['model'])
    optimizer.load_state_dict(state['optimizer'])
    scheduler.load_state_dict(state['scheduler'])
    scaler.load_state_dict(state['scaler'])
    random.setstate(state['rng']['python'])
    np.random.set_state(state['rng']['numpy'])
    torch.set_rng_state(state['rng']['torch'].cpu())
    if state['rng']['cuda']:
        torch.cuda.set_rng_state_all([s.cpu() for s in state['rng']['cuda']])
    dataset.rng.set_state(state['rng']['patches'])
    if state['epoch'] != len(state['history']):
        raise ValueError('Resume history is incomplete')
    return state
