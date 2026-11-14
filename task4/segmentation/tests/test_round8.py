import importlib.util
from pathlib import Path
from types import SimpleNamespace

import numpy as np
import pytest
import torch

from src.config import Config
from src.models import build_model, load_weights, load_checkpoint
from src.losses import build_loss, training_loss
from src.checkpointing import save_resume, load_resume


@pytest.fixture(autouse=True)
def limit_cpu_threads():
    old = torch.get_num_threads()
    torch.set_num_threads(2)
    yield
    torch.set_num_threads(old)


@pytest.mark.parametrize('name', ['fr_unet', 'unet_aux', 'unet_dsconv'])
def test_registered_models_backward_and_checkpoint_roundtrip(name, tmp_path):
    torch.manual_seed(3)
    cfg = Config(model=name, amp=False, aux_warmup_epochs=1)
    model = build_model(cfg)
    optimizer = torch.optim.AdamW(model.parameters(), lr=1e-3)
    x, target = torch.randn(2,3,32,32), torch.zeros(2,1,32,32)
    target[:,:,6:26,15:17] = 1
    extra = {'skeleton': target.clone()}
    for _ in range(2):
        model.train()
        optimizer.zero_grad(set_to_none=True)
        output = model(x)
        loss = training_loss(output, target, build_loss(cfg.loss), cfg, extra)
        loss.backward()
        assert torch.isfinite(loss)
        assert all(torch.isfinite(p.grad).all() for p in model.parameters() if p.grad is not None)
        optimizer.step()
    if name == 'unet_dsconv':
        assert model.enc[0].snake_x.offset_conv.weight.grad.abs().sum()>0
    if name == 'unet_aux':
        assert model.head.skeleton[-1].weight.grad.abs().sum()>0
    model.eval()
    expected = model(x).detach()
    assert expected.shape == target.shape
    path = tmp_path/'candidate.pt'
    torch.save({'model':model.state_dict(),'cfg':cfg.to_dict()},path)
    restored,_,_ = load_checkpoint(path,device='cpu')
    torch.testing.assert_close(restored(x),expected,rtol=0,atol=0)


@pytest.mark.parametrize('name', ['unet_aux','unet_dsconv'])
def test_extended_models_keep_the_pretrained_initial_function(name,tmp_path):
    torch.manual_seed(4)
    baseline = build_model(Config()).eval()
    path = tmp_path/'baseline.pt'
    torch.save({'model':baseline.state_dict()},path)
    extended = build_model(Config(model=name)).eval()
    count,skipped,missing = load_weights(extended,path)
    assert count == len(baseline.state_dict()) and not skipped and missing
    x = torch.randn(2,3,48,48)
    with torch.no_grad():
        torch.testing.assert_close(extended(x),baseline(x),rtol=1e-5,atol=1e-6)


@pytest.mark.parametrize('morph',[0,1])
def test_native_dsconv_matches_upstream_forward_and_gradients(morph,monkeypatch):
    root = Path(__file__).resolve().parents[1]
    monkeypatch.syspath_prepend(str(root/'experiments/hardware_check_20261101/vendor_deps'))
    source = root/'experiments/hardware_check_20261101/vendor/dsconv.py'
    spec = importlib.util.spec_from_file_location('upstream_dsconv_check',source)
    module = importlib.util.module_from_spec(spec)
    spec.loader.exec_module(module)
    from src.vendor.dsconv import DSConv_pro
    upstream = module.DSConv_pro(3,8,kernel_size=5,morph=morph,device='cpu')
    local = DSConv_pro(3,8,kernel_size=5,morph=morph,device='cpu')
    local.load_state_dict(upstream.state_dict())
    x = torch.randn(2,3,16,12,requires_grad=True)
    y = x.detach().clone().requires_grad_(True)
    a,b = upstream(x),local(y)
    torch.testing.assert_close(a,b,rtol=0,atol=0)
    a.square().mean().backward();b.square().mean().backward()
    torch.testing.assert_close(x.grad,y.grad,rtol=0,atol=0)
    torch.testing.assert_close(upstream.offset_conv.weight.grad,local.offset_conv.weight.grad,rtol=0,atol=0)


def test_resume_restores_optimizer_and_all_random_states(tmp_path):
    import random
    torch.manual_seed(12);np.random.seed(12);random.seed(12)
    cfg=Config(out_dir=str(tmp_path),epochs=2,device='cpu',amp=False)
    ds=SimpleNamespace(rng=np.random.RandomState(9))
    def setup():
        model=torch.nn.Sequential(torch.nn.Conv2d(3,1,1),torch.nn.Dropout(.2))
        optimizer=torch.optim.AdamW(model.parameters(),lr=.01)
        scheduler=torch.optim.lr_scheduler.StepLR(optimizer,1,.8)
        scaler=torch.amp.GradScaler('cuda',enabled=False)
        return model,optimizer,scheduler,scaler
    def step(model,opt,scheduler,dataset):
        offset=dataset.rng.rand()+np.random.rand()+random.random()
        x=torch.rand(2,3,8,8)+offset
        opt.zero_grad();model(x).square().mean().backward();opt.step();scheduler.step()
    model,opt,scheduler,scaler=setup()
    step(model,opt,scheduler,ds)
    path=str(tmp_path/'resume.pt')
    save_resume(path,'test',cfg,model,opt,scheduler,scaler,ds,[{'epoch':1}])
    step(model,opt,scheduler,ds)
    expected={k:v.detach().clone() for k,v in model.state_dict().items()}
    model2,opt2,scheduler2,scaler2=setup();ds2=SimpleNamespace(rng=np.random.RandomState(100))
    load_resume(path,'test',cfg,model2,opt2,scheduler2,scaler2,ds2,'cpu')
    step(model2,opt2,scheduler2,ds2)
    for key,value in model2.state_dict().items():
        torch.testing.assert_close(value,expected[key],rtol=0,atol=0)
    cfg.epochs=3
    with pytest.raises(ValueError,match='configuration differs'):
        load_resume(path,'test',cfg,model2,opt2,scheduler2,scaler2,ds2,'cpu')
