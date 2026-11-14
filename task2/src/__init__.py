"""2026task2 灰度人脸六类情感分类：项目包初始化。

约定：
- 内部类别顺序固定为 CLASSES = (Angry, Fear, Happy, Neutral, Sad, Surprise)；
- 竞赛提交编号为 COMP_LABEL = {Angry:0, Fear:1, Happy:2, Sad:3, Surprise:4, Neutral:5}，
  两者顺序不同，写出 CSV 时必须显式转换（见 src/predict.py）。
"""

from __future__ import annotations

import os
from pathlib import Path

ROOT = Path(__file__).resolve().parent.parent
DATA_DIR = ROOT / "fer_data" / "fer_data"
ARTIFACTS = ROOT / "artifacts"
RUNS = ROOT / "outputs" / "runs"
REPORTS = ROOT / "reports"

CLASSES = ("Angry", "Fear", "Happy", "Neutral", "Sad", "Surprise")
NUM_CLASSES = len(CLASSES)
CLASS_TO_IDX = {c: i for i, c in enumerate(CLASSES)}
# 竞赛编号（与 sample_submission 的取值一致）：Angry=0, Fear=1, Happy=2, Sad=3, Surprise=4, Neutral=5
COMP_LABEL = {"Angry": 0, "Fear": 1, "Happy": 2, "Sad": 3, "Surprise": 4, "Neutral": 5}
IDX_TO_COMP = [COMP_LABEL[c] for c in CLASSES]
COMP_TO_IDX = [None] * NUM_CLASSES
for _i, _c in enumerate(IDX_TO_COMP):
    COMP_TO_IDX[_c] = _i

for _d in (ARTIFACTS, RUNS, REPORTS):
    _d.mkdir(parents=True, exist_ok=True)

os.environ.setdefault("PYTHONIOENCODING", "utf-8")
