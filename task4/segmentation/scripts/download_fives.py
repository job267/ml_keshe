"""下载并转换 FIVES 数据集（800 张眼底图 + 血管标注）到 data_external/fives/

FIVES 在 HuggingFace 上有两个入口（`tyluan/FIVES`）：

1. **parquet 原始文件**（4 分片，约 1.75 GB）。本机 pyarrow 19 读这些文件会报
   ``OSError: Repetition level histogram size mismatch``（文件里的
   repetition-level 直方图与 reader 不兼容，`read_table` / `iter_batches` /
   `dataset` 三条路径全部失败），所以默认走下面的方案 2。
2. **datasets-server 的 rows 接口**（默认）：返回每行的签名 URL，
   直接下载 JPEG（image 约 150 KB / mask 约 30 KB），
   实测 mask 阈值化后是干净的二值血管图（2048 尺度血管占比 7.0%），
   总下载量约 150 MB，比 parquet 快得多。

输出：
    data_external/fives/image/fives_<split>_<idx>.png    长边缩放到 --max_side
    data_external/fives/label/fives_<split>_<idx>.png    0/255，血管=255
    data_external/fives/meta_fives.json                  张数/尺寸/血管占比

用法：
    python scripts/download_fives.py                  # 全部 800 张
    python scripts/download_fives.py --limit 300      # 只下前 300 张（调试用）
    python scripts/download_fives.py --source parquet # 换回 parquet（本机不行）
"""
import argparse
import glob
import io
import json
import os
import ssl
import sys
import time
import urllib.parse
import urllib.request
from concurrent.futures import ThreadPoolExecutor, as_completed

import numpy as np
from PIL import Image

ROOT = os.path.dirname(os.path.dirname(os.path.abspath(__file__)))
if ROOT not in sys.path:
    sys.path.insert(0, ROOT)

REPO = "tyluan/FIVES"
RAW_DIR = os.path.join(ROOT, "data_external", "_raw", "fives")
OUT_DIR = os.path.join(ROOT, "data_external", "fives")
API = "https://datasets-server.huggingface.co/rows"

CTX = ssl._create_unverified_context()
UA = {"User-Agent": "Mozilla/5.0"}


def fetch(url, timeout=120, retries=4):
    for k in range(retries):
        try:
            req = urllib.request.Request(url, headers=UA)
            return urllib.request.urlopen(req, timeout=timeout, context=CTX).read()
        except Exception as e:  # noqa: BLE001
            if k == retries - 1:
                raise
            print(f"  [重试 {k+1}] {e}", flush=True)
            time.sleep(2 * (k + 1))


def get_json(url):
    return json.loads(fetch(url, timeout=60))


def list_rows(split, page=100):
    """分页拉取某个 split 的所有行的 URL（datasets-server rows 接口）。"""
    rows = []
    offset = 0
    while True:
        q = urllib.parse.urlencode({"dataset": REPO, "config": "default", "split": split,
                                    "offset": offset, "length": page})
        d = get_json(f"{API}?{q}")
        batch = d.get("rows", [])
        if not batch:
            break
        for r in batch:
            row = r["row"]
            rows.append({
                "image_id": row.get("image_id", f"{split}_{offset + len(rows)}"),
                "image": row["image"]["src"],
                "mask": row["mask"]["src"],
                "size": (row["image"].get("height"), row["image"].get("width")),
            })
        offset += len(batch)
        total = d.get("num_rows_total", offset)
        print(f"  {split}: 已列出 {offset}/{total}", flush=True)
        if offset >= total:
            break
    return rows


def process_one(item, split, idx, max_side):
    name = f"fives_{split}_{idx:04d}"
    ipath = os.path.join(OUT_DIR, "image", name + ".png")
    lpath = os.path.join(OUT_DIR, "label", name + ".png")
    if os.path.exists(ipath) and os.path.exists(lpath):
        return {"id": name, "image_id": item["image_id"], "split": split, "skipped": True}
    img = Image.open(io.BytesIO(fetch(item["image"]))).convert("RGB")
    msk = Image.open(io.BytesIO(fetch(item["mask"]))).convert("L")
    if msk.size != img.size:
        msk = msk.resize(img.size, Image.NEAREST)
    w, h = img.size
    sc = min(1.0, max_side / max(w, h))
    if sc < 1.0:
        nw, nh = int(round(w * sc)), int(round(h * sc))
        img = img.resize((nw, nh), Image.BILINEAR)
        msk = msk.resize((nw, nh), Image.BILINEAR)
    binm = (np.asarray(msk) > 127).astype(np.uint8) * 255
    if binm.mean() > 127.5:  # 极性兜底：白像素过半说明反转了
        binm = 255 - binm
    img.save(ipath, optimize=True)
    Image.fromarray(binm).save(lpath, optimize=True)
    return {"id": name, "image_id": item["image_id"], "split": split,
            "size": [binm.shape[0], binm.shape[1]],
            "vessel_ratio": round(float(binm.mean() / 255), 5)}


def rebuild_meta(max_side):
    """扫描输出目录重建 meta（可随时调用，用于中断后续跑）。"""
    items = []
    for lp in sorted(glob.glob(os.path.join(OUT_DIR, "label", "*.png"))):
        name = os.path.splitext(os.path.basename(lp))[0]
        ip = os.path.join(OUT_DIR, "image", name + ".png")
        if not os.path.exists(ip):
            continue
        m = np.asarray(Image.open(lp).convert("L"))
        items.append({"id": name, "split": name.split("_")[1], "size": list(m.shape),
                      "vessel_ratio": round(float((m > 127).mean()), 5)})
    write_meta(items, max_side)
    print(f"meta 重建完成：{len(items)} 张", flush=True)
    return len(items)


def run_api(max_side=1024, limit=0, workers=8, splits=("test", "train")):
    os.makedirs(os.path.join(OUT_DIR, "image"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "label"), exist_ok=True)
    meta = []
    t0 = time.time()
    for split in splits:
        print(f"[{split}] 列出文件列表 ...", flush=True)
        rows = list_rows(split)
        if limit:
            rows = rows[:limit]
        print(f"[{split}] 下载 {len(rows)} 张", flush=True)
        with ThreadPoolExecutor(max_workers=workers) as ex:
            futs = {ex.submit(process_one, r, split, i, max_side): (r, i)
                    for i, r in enumerate(rows)}
            done = 0
            for f in as_completed(futs):
                r, i = futs[f]
                try:
                    meta.append(f.result())
                except Exception as e:  # noqa: BLE001
                    print(f"  [失败] {r['image_id']}: {e}", flush=True)
                done += 1
                if done % 50 == 0:
                    el = time.time() - t0
                    print(f"    {done}/{len(rows)}  {el:.0f}s", flush=True)
    rebuild_meta(max_side)
    print(f"下载阶段结束，用时 {time.time()-t0:.0f}s", flush=True)
    return len(meta)


def run_parquet(max_side=1024, limit=0):
    """原始 parquet 路径（本机 pyarrow 19 会报错，保留给其他环境）。"""
    import pyarrow.parquet as pq

    files = [f for f in sorted(os.listdir(RAW_DIR)) if f.endswith(".parquet")] if os.path.isdir(RAW_DIR) else []
    if not files:
        raise RuntimeError("没有 parquet 文件，先用 --source api 或手动下载")
    os.makedirs(os.path.join(OUT_DIR, "image"), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, "label"), exist_ok=True)
    meta, n = [], 0
    for f in files:
        split = "test" if f.startswith("test") else "train"
        tbl = pq.read_table(os.path.join(RAW_DIR, f), columns=["image", "mask", "image_id"]).to_pydict()
        for k in range(len(tbl["image"])):
            img = Image.open(io.BytesIO(tbl["image"][k]["bytes"])).convert("RGB")
            msk = Image.open(io.BytesIO(tbl["mask"][k]["bytes"])).convert("L")
            if msk.size != img.size:
                msk = msk.resize(img.size, Image.NEAREST)
            sc = min(1.0, max_side / max(img.size))
            if sc < 1.0:
                img = img.resize((int(img.size[0] * sc), int(img.size[1] * sc)), Image.BILINEAR)
                msk = msk.resize(img.size, Image.BILINEAR)
            binm = (np.asarray(msk) > 127).astype(np.uint8) * 255
            if binm.mean() > 127.5:  # 极性兜底
                binm = 255 - binm
            name = f"fives_{split}_{n:04d}"
            img.save(os.path.join(OUT_DIR, "image", name + ".png"), optimize=True)
            Image.fromarray(binm).save(os.path.join(OUT_DIR, "label", name + ".png"), optimize=True)
            meta.append({"id": name, "image_id": tbl["image_id"][k], "split": split,
                         "size": list(binm.shape), "vessel_ratio": round(float(binm.mean() / 255), 5)})
            n += 1
            if limit and n >= limit:
                break
        if limit and n >= limit:
            break
    write_meta(meta, max_side)
    return n


def write_meta(meta, max_side):
    if not meta:
        return
    r = float(np.mean([m["vessel_ratio"] for m in meta]))
    with open(os.path.join(OUT_DIR, "meta_fives.json"), "w", encoding="utf-8") as f:
        json.dump({"count": len(meta), "max_side": max_side, "vessel_ratio_mean": r,
                   "items": meta}, f, ensure_ascii=False, indent=1)
    print(f"  血管占比均值 {r*100:.2f}%  尺寸样例 {meta[0]['size']}")


def main(argv=None):
    p = argparse.ArgumentParser(description="下载并转换 FIVES 数据集")
    p.add_argument("--source", choices=["api", "parquet"], default="api")
    p.add_argument("--max_side", type=int, default=1024)
    p.add_argument("--limit", type=int, default=0, help="每个 split 只取前 N 张（0=全部）")
    p.add_argument("--workers", type=int, default=8)
    p.add_argument("--rebuild", action="store_true", help="只扫描已下载文件重建 meta_fives.json")
    args = p.parse_args(argv)
    if args.rebuild:
        n = rebuild_meta(args.max_side)
    elif args.source == "api":
        n = run_api(max_side=args.max_side, limit=args.limit, workers=args.workers)
    else:
        n = run_parquet(max_side=args.max_side, limit=args.limit)
    print(f"总计 {n} 张 -> {OUT_DIR}")
    return 0


if __name__ == "__main__":
    raise SystemExit(main())
