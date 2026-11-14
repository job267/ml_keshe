"""提交校验的独立入口（列名、行数、重复 ID、ID 顺序、标签范围）。

    python tools\\validate_submission.py submission_round17_deploy.csv --template submission.csv
"""

from __future__ import annotations

import argparse
import json
import sys
from pathlib import Path

sys.path.insert(0, str(Path(__file__).resolve().parent.parent))

from src.predict import sha256_file, validate_submission  # noqa: E402


def main(argv=None) -> int:
    ap = argparse.ArgumentParser(description="校验提交文件")
    ap.add_argument("csv", help="待校验的提交文件")
    ap.add_argument("--template", default="submission.csv", help="sample_submission（用于核对 ID 顺序）")
    args = ap.parse_args(argv)
    tpl = args.template if Path(args.template).exists() else None
    rep = validate_submission(args.csv, tpl)
    rep["sha256"] = sha256_file(args.csv)
    print(json.dumps(rep, ensure_ascii=False, indent=2))
    return 0 if rep["ok"] else 1


if __name__ == "__main__":
    raise SystemExit(main())
