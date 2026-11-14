"""眼底血管分割 —— 课程设计基础框架。

模块划分::

    config.py      超参数集中管理（支持命令行覆盖）
    utils.py       随机种子、设备、图像读写、FOV 提取等通用工具
    preprocess.py  绿色通道 / CLAHE / 高通 三通道构造
    augment.py     训练期数据增强（几何 + 光度）
    dataset.py     Patch 训练集与整图验证集
    models.py      模型注册表（U-Net / ResU-Net ...）
    losses.py      损失函数（BCE+Dice / Focal+Dice）
    metrics.py     Dice/SE/SP/ACC 等评价指标
    postprocess.py 阈值化、FOV 掩膜、小连通域剔除
    engine.py      前向推理（含 TTA）与权重加载
    train.py       训练入口
    infer.py       测试集推理入口（输出 PNG 预测图）
    rle.py         提交 CSV 生成 / 校验（与官方脚本等价）
    evaluate.py    预测结果评估与可视化
    frangi.py      传统方法基线（Frangi 滤波）
"""

__version__ = "1.0.0"
