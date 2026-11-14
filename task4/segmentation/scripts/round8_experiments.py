"""Staged Round8 architecture experiments. Serial GPU execution; fixed final epochs."""

import argparse
import contextlib
import dataclasses
import gc
import hashlib
import json
from pathlib import Path
import shutil
import sys
import time

ROOT = Path(__file__).resolve().parents[1]
sys.path.insert(0,str(ROOT))

import numpy as np
import torch
from src.config import Config
from src.cv import stratified_folds,read_folds
from src.dataset import build_records,build_external_records
from src.train import main as train_main
from src.pretrain import main as pretrain_main
from scripts.round7_experiments import Tee
from scripts.round7_audit import (cache_oof,ground_truth,paired,score_probs,sha256,
                                  source_hashes,write_csv,write_json)


MODELS={'baseline':'unet','fr_unet':'fr_unet','auxiliary':'unet_aux','dsconv':'unet_dsconv'}


def digest(data):
    return hashlib.sha256(json.dumps(data,sort_keys=True).encode()).hexdigest()


def run_logged(call,cfg,path):
    path=Path(path);path.parent.mkdir(parents=True,exist_ok=True)
    with path.open('a',encoding='utf-8') as log:
        with contextlib.redirect_stdout(Tee(sys.stdout,log)):
            call([f'--{key}={value}' for key,value in cfg.to_dict().items()])
    gc.collect();torch.cuda.empty_cache()


class Experiment:
    def __init__(self,args):
        self.args=args;self.out=Path(args.out_dir);self.out.mkdir(parents=True,exist_ok=True)
        self.records=build_records('train/image','train/label')
        self.folds=stratified_folds(self.records,5,seed=args.split_seed)
        self.split={f'fold{k}':ids for k,ids in enumerate(self.folds)}
        external=build_external_records(['data_external/stare','data_external/chase','data_external/hrf'])
        self.plan={'source':source_hashes(),'runner_sha256':sha256(__file__),
                   'split_seed':args.split_seed,'split':self.split,'epochs':args.epochs,
                   'screen_folds':[0,1],'confirmation_folds':[2,3,4],
                   'training_seeds':[42,143],'models':MODELS,'checkpoint_policy':'last',
                   'pretraining':'FR-UNet: same 93 external images, 40 epochs x1200 patches; U-Net body reuses existing 40-epoch checkpoint',
                   'baseline_pretrain_sha256':sha256('outputs/pretrained.pt'),
                   'data':{r['id']:[sha256(r['image']),sha256(r['label'])] for r in self.records},
                   'external_data':{r['id']:[sha256(r['image']),sha256(r['label'])] for r in external},
                   'target_recipe':{'lr':5e-4,'batch':8,'patch':192,'patches_per_epoch':480,
                                    'strong_aug':True,'cache_dtype':'float16','tta':True,'threshold':.45,'min_size':10},
                   'auxiliary':'BCE+Dice for region, boundary and skeleton; each auxiliary weight ramps to 0.05 over 10 epochs',
                   'dsconv':'native-PyTorch equivalent operator, identity-initialized first-stage fusion',
                   'screen_gate':{'proxy_delta_min':.002,'native_delta_min':0.,'both_folds_positive':True,'image_wins_min':5},
                   'remaining_folds_gate':{'proxy_delta_min':.001,'native_delta_min':0.},
                   'second_seed_gate':{'proxy_delta_min':.001,'native_delta_min':0.},
                   'hidden_gt_bilinear':'unconfirmed proxy; native and thin/precision metrics also reported'}
        planfile=self.out/'plan.json'
        if planfile.exists() and json.loads(planfile.read_text(encoding='utf-8'))!=self.plan:
            raise RuntimeError('Round8 source/data/protocol changed; use the frozen source or another output directory')
        write_json(planfile,self.plan)
        write_json(self.out/'split.json',{'split':self.split,'split_seed':args.split_seed})
        self.truth=ground_truth(self.records)
        self.fold_by_id={rid:k for k,ids in enumerate(self.folds) for rid in ids}

    def status(self,stage,**details):
        write_json(self.out/'status.json',{'stage':stage,**details})

    def pretrain(self):
        folder=self.out/'pretrain/fr_unet';folder.mkdir(parents=True,exist_ok=True)
        cfg=Config(model='fr_unet',out_dir=str(folder),pretrain_out=str(folder/'pretrained.pt'),
                   external_dirs='data_external/stare,data_external/chase,data_external/hrf',
                   epochs=40,patches_per_epoch=1200,batch_size=8,lr=1e-3,
                   val_ids='',cache_dtype='float16',device='cuda',checkpoint_every=5)
        signature=digest({'cfg':cfg.to_dict(),'source':self.plan['source'],
                          'data':self.plan['external_data']})
        marker=folder/'completed.json'
        if marker.exists():
            done=json.loads(marker.read_text(encoding='utf-8'))
            if done['signature']!=signature or done['sha256']!=sha256(cfg.pretrain_out):
                raise ValueError('Pretraining provenance mismatch')
            return cfg.pretrain_out
        if Path(cfg.pretrain_out).exists():
            ck=torch.load(cfg.pretrain_out,map_location='cpu',weights_only=False)
            if (ck['cfg']['model']!='fr_unet' or len(ck['meta']['history'])!=40 or ck['meta']['records']!=93):
                raise ValueError('Unverified pretraining checkpoint')
        else:
            self.status('external_pretraining',model='fr_unet',epochs=40)
            actual=dataclasses.replace(cfg,resume_from=str(folder/'resume.pt') if (folder/'resume.pt').exists() else '')
            run_logged(pretrain_main,actual,folder/'console.log')
        write_json(marker,{'signature':signature,'sha256':sha256(cfg.pretrain_out),'cfg':cfg.to_dict()})
        return cfg.pretrain_out

    def train_fold(self,tag,fold,seed,initializer):
        run=self.out/f'seed{seed}'/tag;folder=run/f'fold{fold}'
        cfg=Config(model=MODELS[tag],out_dir=str(folder),epochs=self.args.epochs,
                   seed=seed+fold,split_seed=self.args.split_seed,split_strategy='saved',
                   val_ids=','.join(self.folds[fold]),init_from=initializer,
                   strong_aug=True,cache_dtype='float16',lr=5e-4,batch_size=8,
                   val_every=20,selection_tta=True,selection_metric='dice_512_proxy',
                   checkpoint_policy='last',device='cuda',checkpoint_every=10)
        signature=digest({'cfg':cfg.to_dict(),'initializer':sha256(initializer),'plan':digest(self.plan)})
        marker=folder/'completed.json'
        if marker.exists():
            done=json.loads(marker.read_text(encoding='utf-8'))
            if done['signature']!=signature or done['sha256']!=sha256(folder/'best.pt'):
                raise ValueError(f'Completed fold provenance mismatch: {folder}')
            print('REUSE',folder,flush=True)
            return
        if (folder/'best.pt').exists() and not (folder/'resume.pt').exists():
            raise RuntimeError(f'Incomplete fold without resume state; preserve it and use another output folder: {folder}')
        self.status('target_training',model=tag,fold=fold,seed=seed,epochs=cfg.epochs)
        started=time.monotonic();print('BEGIN',tag,'fold',fold,'seed',seed,flush=True)
        actual=dataclasses.replace(cfg,resume_from=str(folder/'resume.pt') if (folder/'resume.pt').exists() else '')
        run_logged(train_main,actual,folder/'console.log')
        write_json(marker,{'signature':signature,'sha256':sha256(folder/'best.pt'),
                           'cfg':cfg.to_dict(),'seconds_this_session':time.monotonic()-started})

    def ensure(self,tag,folds,seed,initializer):
        for fold in folds:self.train_fold(tag,fold,seed,initializer)
        run=self.out/f'seed{seed}'/tag
        completed=[k for k in range(5) if (run/f'fold{k}/completed.json').exists()]
        write_json(run/'cv_summary.json',{'split':self.split,'evaluated_folds':completed,
                                         'plan':digest(self.plan),'checkpoint_policy':'last'})
        return run

    def evaluate(self,tags,seed,label,ids=None):
        dest=self.out/f'seed{seed}'/f'evaluation_{label}'
        self.status('evaluation',seed=seed,label=label,models=tags)
        details=[];summaries=[];by={}
        for tag in tags:
            probs=cache_oof(self.out/f'seed{seed}'/tag,'checkpoint',self.out/'probability_cache',self.records,torch.device('cuda'))
            if ids is not None:probs={i:p for i,p in probs.items() if i in ids}
            rows,summary=score_probs(tag,probs,self.truth)
            details.extend(rows);summaries.append(summary);by[tag]=rows
            print('RESULT',label,'seed',seed,tag,json.dumps(summary),flush=True)
        write_csv(dest/'per_image.csv',details);write_csv(dest/'summary.csv',summaries)
        comparisons=[paired(rows,by['baseline'],self.fold_by_id,metric) for tag,rows in by.items() if tag!='baseline'
                     for metric in ['dice_512_proxy','dice','precision','thin_recall','skeleton_recall']]
        if comparisons:write_csv(dest/'paired.csv',comparisons)
        return by,summaries

    def run(self):
        initializer=self.pretrain()
        init={tag:initializer if tag=='fr_unet' else 'outputs/pretrained.pt' for tag in MODELS}
        for tag in MODELS:self.ensure(tag,[0,1],42,init[tag])
        by,summary=self.evaluate(list(MODELS),42,'screen')
        promoted=[];decisions={}
        for tag in list(MODELS)[1:]:
            p=paired(by[tag],by['baseline'],self.fold_by_id)
            native=paired(by[tag],by['baseline'],self.fold_by_id,'dice')
            passed=p['mean_delta']>=.002 and native['mean_delta']>=0 and p['fold_wins']==2 and p['image_wins']>=5
            decisions[tag]={'screen_passed':passed,'proxy':p,'native':native}
            if passed:promoted.append(tag)
        write_json(self.out/'screen_decision.json',decisions)
        confirmed=[]
        remaining={i for k in [2,3,4] for i in self.folds[k]}
        if promoted:self.ensure('baseline',[2,3,4],42,init['baseline'])
        for tag in promoted:
            self.ensure(tag,[2,3,4],42,init[tag])
            hold,_=self.evaluate(['baseline',tag],42,f'heldout_{tag}',remaining)
            p=paired(hold[tag],hold['baseline'],self.fold_by_id)
            n=paired(hold[tag],hold['baseline'],self.fold_by_id,'dice')
            full,_=self.evaluate(['baseline',tag],42,f'full_{tag}')
            full_delta=paired(full[tag],full['baseline'],self.fold_by_id)['mean_delta']
            decisions[tag]['heldout']={'proxy':p,'native':n,'full_proxy_delta':full_delta}
            if p['mean_delta']>=.001 and n['mean_delta']>=0 and full_delta>=.002:
                self.ensure('baseline',range(5),143,init['baseline'])
                self.ensure(tag,range(5),143,init[tag])
                second,_=self.evaluate(['baseline',tag],143,f'full_{tag}')
                pp=paired(second[tag],second['baseline'],self.fold_by_id)
                nn=paired(second[tag],second['baseline'],self.fold_by_id,'dice')
                decisions[tag]['second_seed']={'proxy':pp,'native':nn}
                if pp['mean_delta']>=.001 and nn['mean_delta']>=0:confirmed.append(tag)
        candidates=[]
        if confirmed:
            from scripts.ensemble import main as ensemble_main
            from src.rle import build_submission,validate_rle
            from segmentation_to_csv import turn_to_str
            for tag in confirmed:
                self.status('candidate_generation',model=tag)
                dest=self.out/f'candidate_{tag}'
                image_dir=dest/'image'
                checkpoints=[str(self.out/f'seed{seed}'/tag/f'fold{k}/best.pt') for seed in [42,143] for k in range(5)]
                ensemble_main(['--checkpoints',','.join(checkpoints),'--data_dir','test',
                               '--pred_dir',str(image_dir),'--threshold','.45','--min_size','10'])
                csv_path=dest/f'submission_round8_{tag}.csv'
                table=build_submission(str(image_dir),str(csv_path))
                if len(table)!=20 or not all(validate_rle(s)['ok'] for s in table['Predicted']):
                    raise ValueError('Candidate submission validation failed')
                paths=[str(image_dir/f'{rid}.png') for rid in table['Id']]
                if turn_to_str(paths)!=table['Predicted'].tolist():
                    raise ValueError('Official encoder and candidate RLE differ')
                write_json(dest/'manifest.json',{'model':tag,'csv_sha256':sha256(csv_path),
                                                 'checkpoints':{p:sha256(p) for p in checkpoints},
                                                 'plan':digest(self.plan),'threshold':.45,'min_size':10,'tta':True,
                                                 'submitted_to_kaggle':False})
                candidates.append(str(csv_path))
        write_json(self.out/'decision.json',{'decisions':decisions,'confirmed_candidates':confirmed,
                                           'new_submission_generated':bool(candidates),'candidate_files':candidates,
                                           'note':'Only candidates passing remaining-fold and second-seed confirmation are exported; scores are local proxies.'})
        self.status('complete',confirmed_candidates=confirmed)


def main():
    parser=argparse.ArgumentParser();parser.add_argument('--out_dir',default='experiments/round8')
    parser.add_argument('--epochs',type=int,default=60);parser.add_argument('--split_seed',type=int,default=20261101)
    args=parser.parse_args();exp=Experiment(args)
    snapshot=exp.out/'implementation'
    if not snapshot.exists():
        for top in ['src','scripts','tests']:
            for p in (ROOT/top).rglob('*'):
                if p.is_file() and p.suffix in ['.py','.json'] or p.is_file() and p.name.endswith('_LICENSE'):
                    q=snapshot/p.relative_to(ROOT);q.parent.mkdir(parents=True,exist_ok=True);shutil.copy2(p,q)
    exp.run()


if __name__=='__main__':main()
