"""下载外部眼底血管分割数据集（STARE / CHASE_DB1 / HRF）。

数据来源：Hugging Face 上的公开镜像（无需注册、可直接下载）
    Zomba/STARE-structured-analysis-of-the-retina   20 张 + 手工标注
    Zomba/CHASE_DB1-retinal-dataset                 28 张 + 2 位观察者标注
    Zomba/HRF-high-resolution-fundus                45 张（健康/糖网/青光眼各 15）+ 标注

下载到 ``data_external/_raw/<数据集>/<split>/<input|label>/``，
再由 ``scripts/prepare_external.py`` 整理成统一的 ``image/ + label/`` 结构。

用法::

    python scripts/download_external.py                 # 全部
    python scripts/download_external.py --only stare chase
    python scripts/download_external.py --workers 8
"""

import argparse
import json
import os
import ssl
import sys
import time
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
RAW = os.path.join(ROOT, "data_external", "_raw")

DATASETS = {
    "stare": {
        "repo": "Zomba/STARE-structured-analysis-of-the-retina",
        "note": "STARE, 20 张 700x605, .ppm 图 + .ah.ppm 金标准",
    },
    "chase": {
        "repo": "Zomba/CHASE_DB1-retinal-dataset",
        "note": "CHASE_DB1, 28 张 999x960, png + _1stHO.png 标注",
    },
    "hrf": {
        "repo": "Zomba/HRF-high-resolution-fundus",
        "note": "HRF, 45 张 3504x2336, JPG + tif 标注",
    },
}

SKIP_NAMES = {".gitattributes", "README.md", "thumbs.db", ".DS_Store"}
CTX = ssl._create_unverified_context()   # 部分镜像站点证书链不完整


def api_json(url, retries=3):
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            return json.loads(urllib.request.urlopen(req, timeout=40, context=CTX).read())
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (k + 1))
    raise RuntimeError(f"接口请求失败 {url}: {last}")


def list_files(repo):
    info = api_json(f"https://huggingface.co/api/datasets/{repo}")
    out = []
    for s in info.get("siblings", []):
        path = s["rfilename"]
        name = os.path.basename(path)
        if name in SKIP_NAMES or path.endswith("/"):
            continue
        if not (path.startswith("train/") or path.startswith("val/")):
            continue
        out.append(path)
    return sorted(out)


def download(repo, path, dst, retries=3):
    if os.path.exists(dst) and os.path.getsize(dst) > 0:
        return "skip", os.path.getsize(dst)
    os.makedirs(os.path.dirname(dst), exist_ok=True)
    url = f"https://huggingface.co/datasets/{repo}/resolve/main/{path}"
    last = None
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers={"User-Agent": "Mozilla/5.0"})
            data = urllib.request.urlopen(req, timeout=120, context=CTX).read()
            if not data:
                raise RuntimeError("空文件")
            tmp = dst + ".part"
            with open(tmp, "wb") as f:
                f.write(data)
            os.replace(tmp, dst)
            return "ok", len(data)
        except Exception as e:  # noqa: BLE001
            last = e
            time.sleep(1.5 * (k + 1))
    return f"fail: {last}", 0


def main(argv=None):
    p = argparse.ArgumentParser(description="下载外部数据集")
    p.add_argument("--only", nargs="*", default=None, help="只下指定数据集: stare chase hrf")
    p.add_argument("--workers", type=int, default=8)
    args = p.parse_args(argv)

    keys = list(DATASETS) if not args.only else args.only
    total_ok = total_skip = total_bytes = 0
    failures = []

    for key in keys:
        if key not in DATASETS:
            print(f"[跳过] 未知数据集 {key}")
            continue
        repo = DATASETS[key]["repo"]
        print(f"\n=== {key}  ({DATASETS[key]['note']})")
        try:
            files = list_files(repo)
        except Exception as e:  # noqa: BLE001
            print(f"  列出文件失败: {e}")
            failures.append(key)
            continue
        print(f"  待下载 {len(files)} 个文件 -> {os.path.join(RAW, key)}")

        with ThreadPoolExecutor(max_workers=args.workers) as ex:
            futs = {ex.submit(download, repo, path, os.path.join(RAW, key, path)): path
                    for path in files}
            for fut in as_completed(futs):
                status, n = fut.result()
                path = futs[fut]
                if status == "ok":
                    total_ok += 1
                    total_bytes += n
                elif status == "skip":
                    total_skip += 1
                    total_bytes += n
                else:
                    failures.append(f"{key}/{path}: {status}")
                    print(f"  [失败] {path} -> {status}")

    print("\n" + "-" * 70)
    print(f"新下载 {total_ok} 个，已存在 {total_skip} 个，合计 {total_bytes/1e6:.1f} MB")
    if failures:
        print(f"失败 {len(failures)} 项：")
        for f in failures[:10]:
            print("   ", f)
        return 1
    print("全部就绪。下一步：python scripts/prepare_external.py")
    return 0


if __name__ == "__main__":
    sys.exit(main())
