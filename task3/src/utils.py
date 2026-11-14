"""通用工具：随机种子、日志、EMA、检查点读写。"""

from __future__ import annotations

import json
import os
import random
import time
from pathlib import Path
from typing import Any, Dict, Optional

import numpy as np
import torch

from . import ROOT  # noqa: F401  (设置 TORCH_HOME 的副作用)


# --------------------------------------------------------------------------- #
# 随机性
# --------------------------------------------------------------------------- #
def set_seed(seed: int = 42, deterministic: bool = False) -> None:
    random.seed(seed)
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    torch.backends.cudnn.benchmark = not deterministic
    if deterministic:
        torch.backends.cudnn.deterministic = True


def worker_init_fn(worker_id: int) -> None:
    """DataLoader worker 的随机种子，保证可复现。"""
    base = torch.initial_seed() % 2**31
    np.random.seed(base + worker_id)
    random.seed(base + worker_id)


# --------------------------------------------------------------------------- #
# 路径 / 日志
# --------------------------------------------------------------------------- #
def resolve_path(p: str | Path) -> Path:
    """相对路径按项目根目录解析。"""
    p = Path(p)
    return p if p.is_absolute() else (ROOT / p)


class Logger:
    """同时输出到控制台和日志文件。"""

    def __init__(self, path: Optional[str | Path] = None):
        self.path = resolve_path(path) if path else None
        if self.path:
            self.path.parent.mkdir(parents=True, exist_ok=True)
            self.path.write_text("", encoding="utf-8")

    def __call__(self, msg: str = "") -> None:
        line = f"[{time.strftime('%H:%M:%S')}] {msg}"
        print(line, flush=True)
        if self.path:
            with self.path.open("a", encoding="utf-8") as f:
                f.write(line + "\n")


class AverageMeter:
    def __init__(self):
        self.sum = 0.0
        self.count = 0

    def update(self, v: float, n: int = 1) -> None:
        self.sum += float(v) * n
        self.count += n

    @property
    def avg(self) -> float:
        return self.sum / max(self.count, 1)


# --------------------------------------------------------------------------- #
# 模型权重滑动平均（EMA）：小数据集上通常能稳定提升 1~3 px
# --------------------------------------------------------------------------- #
class ModelEma:
    def __init__(self, model: torch.nn.Module, decay: float = 0.999):
        self.decay = decay
        self.num_updates = 0
        self.ema: Dict[str, torch.Tensor] = {
            k: v.detach().clone() for k, v in model.state_dict().items()
        }

    @torch.no_grad()
    def update(self, model: torch.nn.Module) -> None:
        self.num_updates += 1
        # 训练初期让 EMA 快速跟上
        d = min(self.decay, (1.0 + self.num_updates) / (10.0 + self.num_updates))
        for k, v in model.state_dict().items():
            if v.dtype.is_floating_point:
                self.ema[k].mul_(d).add_(v.detach(), alpha=1.0 - d)
            else:
                self.ema[k] = v.detach().clone()

    def state_dict(self) -> Dict[str, torch.Tensor]:
        return self.ema

    def load_into(self, model: torch.nn.Module) -> None:
        model.load_state_dict(self.ema, strict=True)


# --------------------------------------------------------------------------- #
# 检查点
# --------------------------------------------------------------------------- #
def save_checkpoint(path: str | Path, payload: Dict[str, Any]) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    torch.save(payload, path)


def load_checkpoint(path: str | Path, map_location="cpu") -> Dict[str, Any]:
    return torch.load(resolve_path(path), map_location=map_location, weights_only=False)


def save_json(path: str | Path, obj: Any) -> None:
    path = resolve_path(path)
    path.parent.mkdir(parents=True, exist_ok=True)
    path.write_text(json.dumps(obj, ensure_ascii=False, indent=2), encoding="utf-8")


def load_json(path: str | Path) -> Any:
    return json.loads(resolve_path(path).read_text(encoding="utf-8"))


def count_params(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


def format_seconds(s: float) -> str:
    s = int(s)
    h, rem = divmod(s, 3600)
    m, sec = divmod(rem, 60)
    return f"{h:d}h{m:02d}m{sec:02d}s" if h else f"{m:d}m{sec:02d}s"
