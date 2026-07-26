# M4-SAR

面向光学影像与 SAR 影像融合目标检测的实验仓库，基于 M4-SAR 数据集及 Ultralytics 检测框架。仓库包含数据处理、训练、测试、预测可视化和热力图生成代码，并在原有方法基础上加入了 **IRDFUSION** 与 **MS2FUSION**。

## 数据集

M4-SAR 包含多分辨率、多极化、多场景和多源影像，图像尺寸为 `512 × 512`，共 6 个类别。数据下载地址：

- [Kaggle](https://kaggle.com/datasets/a8ca500cbad658d8ae1af3d1f84566a5b4e94fe0ddb0be801c9e2f672db36a57)
- [Baidu 网盘](https://pan.baidu.com/s/14iuaf_2ymzpP68EJY0dUyg?pwd=0601)
- [Hugging Face](https://huggingface.co/datasets/wchao0601/m4-sar)

数据集的光学分辨率与 SAR 极化信息如下：

| 文件范围 | 光学分辨率 | SAR 极化 |
| --- | --- | --- |
| `1.jpg ~ 56087.jpg` | 10 m | VH |
| `56088.jpg ~ 112174.jpg` | 60 m | VV |

## 方法

除 M4-SAR 原有的融合模型外，本仓库新增：

- **IRDFUSION**：基于迭代差分 Transformer 的跨模态特征交互与融合方法。
- **MS2FUSION**：基于 SSF 等模块实现的多尺度跨模态特征融合方法。

对应网络模块位于 `ultralytics/nn/modules/`，模型配置位于 `ultralytics/cfg/models/`。可根据实验需要选择相应 YAML 配置进行训练。

## 环境安装

```bash
conda create -n m4-sar python=3.11
conda activate m4-sar

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
  --index-url https://download.pytorch.org/whl/cu118
pip install seaborn thop timm einops
pip install -r requirements.txt
```

项目中的 `STTrack/mamba_install/` 包含部分需要本地编译安装的依赖；如果使用对应 Mamba 模块，请先安装 `causal-conv1d` 和 `selective_scan`。

## 使用方式

在 `train.py` 中设置数据集、模型配置和设备后运行：

```bash
# 单 GPU
python train.py

# 多 GPU：先在 multigpu-train.py 中设置 device
python multigpu-train.py

# 测试
python test.py
```

预测标签、预测结果可视化和热力图：

```bash
python gen-predict-label.py
python vis-predict-label.py
python gen-heatmap.py
```

训练入口和配置中的路径需要根据本地数据位置进行调整。已有训练结果保存在 `runs/` 目录中。

## 项目结构

```text
M4-SAR/
├── ultralytics/              # 检测框架及融合模块
├── ultralytics/cfg/models/   # 模型配置
├── STTrack/                  # Mamba 相关依赖
├── train.py                  # 单 GPU 训练
├── multigpu-train.py         # 多 GPU 训练
├── test.py                   # 测试
└── runs/                     # 训练输出
```

## 引用

如果本项目或 M4-SAR 数据集对你的研究有帮助，请引用：

```bibtex
@article{wang2025m4,
  title={M4-SAR: A Multi-Resolution, Multi-Polarization, Multi-Scene, Multi-Source Dataset and Benchmark for Optical-SAR Fusion Object Detection},
  author={Wang, Chao and Lu, Wei and Li, Xiang and Yang, Jian and Luo, Lei},
  journal={arXiv preprint arXiv:2505.10931},
  year={2025}
}
```

