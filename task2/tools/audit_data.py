"""数据审计：生成 artifacts/metadata.json（报告 E02 的对应记录）。

内容：类别分布、重复/镜像组数、标签冲突组、常量与低对比度图、与测试集的共享像素、
以及第一轮分组分层划分的实际张数。

    python tools\\audit_data.py --out artifacts/metadata.json
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src import ARTIFACTS, DATA_DIR, ROOT  # noqa: E402
from src.config import load_cfg  # noqa: E402
from src.data import audit_summary, list_test, prepare_data  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="数据审计")
    ap.add_argument("--root", default=str(DATA_DIR))
    ap.add_argument("--out", default=str(ARTIFACTS / "metadata.json"))
    ap.add_argument("--skip-test", action="store_true", help="跳过与测试集的像素比对（较慢）")
    args = ap.parse_args(argv)

    cfg = load_cfg(None)
    cfg["data"]["root"] = args.root
    samples, groups, splits, _w, meta = prepare_data(cfg)
    test_paths = [] if args.skip_test else list_test(args.root)
    summary = audit_summary(samples, groups, test_paths)

    out = {
        "root": str(Path(args.root)),
        "n_test": len(test_paths),
        # 组级明细留在 artifacts/dup_index.json（缓存），这里只保留汇总
        "duplicates": {k: v for k, v in meta["dup"].items() if k != "group_sizes"},
        "audit": summary,
        "split": {k: len(v) for k, v in splits.items()},
        "split_per_class": {
            s: {c: int(sum(1 for i in splits[s] if samples[i][1] == j))
                for j, c in enumerate(summary["classes"])}
            for s in splits
        },
    }
    p = Path(args.out)
    p.parent.mkdir(parents=True, exist_ok=True)
    p.write_text(json.dumps(out, ensure_ascii=False, indent=2), encoding="utf-8")
    print(json.dumps({k: out[k] for k in ("n_test", "split")}, ensure_ascii=False, indent=2))
    print(f"已写出 {p}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
