# 机器学习课程设计

四项图像学习实验，涵盖图像分类、关键点定位和像素级分割。

| 目录 | 任务 | 主要方法 |
| --- | --- | --- |
| [`task1/`](task1/) | 植物幼苗分类 | ResNet18 基线、EfficientNet、五折验证和模型融合 |
| [`task2/`](task2/) | 灰度人脸表情分类 | 残差网络、迁移学习和多模型融合 |
| [`task3/`](task3/) | 眼底中心凹定位 | 整图热力图粗定位和局部精定位 |
| [`task4/segmentation/`](task4/segmentation/) | 眼底血管分割 | U-Net、边界与骨架辅助训练、掩膜编码 |

## 环境准备

建议使用 Python 3.10–3.12，为各任务分别创建虚拟环境。在相应任务目录运行 `python -m pip install -r requirements.txt`。Task 4 的依赖文件和运行入口位于 `task4/segmentation/`。使用 GPU 时，安装与你的 CUDA 环境匹配的 PyTorch 和 torchvision。

运行前请在本地准备课程数据、提交模板及所选方案需要的权重。以下数据路径均相对于对应任务的工作目录。

## Task 1 植物幼苗分类

在 `task1/` 下，将训练图像按类别放入 `data/train/`，测试图像放入 `data/test/`，并将提交模板放在 `data/submission-for-task2.csv`。

在 Task 1 的环境中安装 JupyterLab（`python -m pip install jupyterlab`），然后从 `task1/` 启动 JupyterLab 并打开 [`plant_classification.ipynb`](task1/plant_classification.ipynb)。先运行首个配置单元，再运行所选实验需要的代码单元。可以通过 `TASK1_DATA_DIR`、`TASK1_SAMPLE_SUBMISSION` 和 `TASK1_OUTPUT_DIR` 指定数据、模板及输出位置。默认输出目录为 `outputs/`。

## Task 2 灰度人脸表情分类

在 `task2/` 下，训练图像按类别存放于 `fer_data/fer_data/train/`，测试图像存放于 `fer_data/fer_data/test/`。预测时可用 `--template` 指定课程提供的提交模板。

```bash
python -m src.train --config configs/round1_baseline.yaml --out-dir outputs/runs/r1_baseline
python -m src.predict --runs outputs/runs/r1_baseline --template submission.csv --out submission_baseline.csv
```

[`configs/`](task2/configs/) 包含各轮训练和融合配置。采用人脸领域预训练的方案时，在相应配置中通过 `model.pretrained_path` 指定本地权重。

## Task 3 眼底中心凹定位

在 `task3/` 下，图像放入 `detection/train/` 和 `detection/test/`，坐标标签和提交模板分别放在 `detection/fovea_localization_train_GT.csv` 和 `detection/sample_submission.csv`。

以下命令以第 0 折为例。第二阶段需要同一折第一阶段的权重。

```bash
python -m src.train --config configs/stage1.yaml --stage 1 --fold 0
python -m src.train --config configs/stage2.yaml --stage 2 --fold 0 --s1-ckpt outputs/runs/s1_fold0/best.pt
python -m src.predict --config configs/stage2.yaml --runs outputs/runs/s2_fold0 --split test --out outputs/submission.csv
```

在 PowerShell 中可运行 `./scripts/run_all_folds.ps1`，依次训练两阶段的五折模型并生成预测文件。

## Task 4 眼底血管分割

在 `task4/segmentation/` 下，将原图、血管掩膜和测试图像分别放入 `train/image/`、`train/label/` 和 `test/image/`。原图与掩膜通过文件名（不含扩展名）配对，标签中血管为 255，背景为 0。

```bash
python -m src.train --epochs 60
python -m src.infer --checkpoint outputs/best.pt
python segmentation_to_csv.py
```

推理默认将掩膜写入 `image/`，编码脚本生成 `submission.csv`。第八轮模型比较入口为 [`scripts/round8_experiments.py`](task4/segmentation/scripts/round8_experiments.py)，复用历史训练的脚本需要对应的本地检查点及实验记录。

FR-UNet 和 DSCNet 的来源与许可见 [`src/vendor/`](task4/segmentation/src/vendor/)。
