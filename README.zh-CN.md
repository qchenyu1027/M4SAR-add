# M4-SAR

面向光学影像与 SAR 影像融合目标检测的实验仓库，基于 M4-SAR 数据集和 Ultralytics 检测框架。仓库提供训练、测试、预测可视化及热力图生成代码，并集成了 ICAFusion、IRDFusion 和 MS2Fusion 三种融合方法。

## 数据集

M4-SAR 包含多分辨率、多极化、多场景和多源影像，图像尺寸为 `512 × 512`，共 6 个类别。

- [Kaggle](https://kaggle.com/datasets/a8ca500cbad658d8ae1af3d1f84566a5b4e94fe0ddb0be801c9e2f672db36a57)
- [Baidu 网盘](https://pan.baidu.com/s/14iuaf_2ymzpP68EJY0dUyg?pwd=0601)
- [Hugging Face](https://huggingface.co/datasets/wchao0601/m4-sar)

| 文件范围 | 光学分辨率 | SAR 极化 |
| --- | --- | --- |
| `1.jpg ~ 56087.jpg` | 10 m | VH |
| `56088.jpg ~ 112174.jpg` | 60 m | VV |

## 融合方法

| 方法 | 简介 | 参考项目 |
| --- | --- | --- |
| **ICAFusion** | 基于迭代交叉注意力的跨模态特征融合方法。 | [ICAFusion](https://github.com/chanchanchan97/ICAFusion) |
| **IRDFusion** | 基于迭代差分 Transformer 的跨模态特征交互与融合方法。 | [IRDFusion](https://github.com/61s61min/IRDFusion) |
| **MS2Fusion** | 基于 SSF 等模块的多尺度跨模态特征融合方法。 | [MS2Fusion](https://github.com/61s61min/MS2Fusion) |

本仓库中的自定义融合模块位于 `ultralytics/nn/modules/`。

## 模型配置

对应的 YOLO11 OBB 配置文件如下：

```text
ultralytics/cfg/models/benchmark/yolo11-obb-ICAFusion.yaml
ultralytics/cfg/models/benchmark/yolo11-obb-IRDFusion.yaml
ultralytics/cfg/models/benchmark/yolo11-obb-MS2FUSION.yaml
```

训练时将相应 YAML 文件传入模型配置参数即可。配置中的输入为光学影像与 SAR 影像双模态数据，类别数为 6。

## 环境安装

```bash
conda create -n m4-sar python=3.11
conda activate m4-sar

pip install torch==2.1.1 torchvision==0.16.1 torchaudio==2.1.1 \
  --index-url https://download.pytorch.org/whl/cu118
pip install seaborn thop timm einops
pip install -r requirements.txt
```

如果使用 Mamba 相关模块，请先安装 `STTrack/mamba_install/` 中的 `causal-conv1d` 和 `selective_scan`。

## 使用方式

根据实验需求在训练脚本中设置数据集、模型配置和设备：

```bash
# 单 GPU
python train.py

# 多 GPU
python multigpu-train.py

# 测试
python test.py
```

其他工具：

```bash
python gen-predict-label.py
python vis-predict-label.py
python gen-heatmap.py
```

训练输出默认保存在 `runs/` 目录中，数据集和权重路径请根据本地环境调整。

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

```bibtex
@article{wang2025m4,
  title={M4-SAR: A Multi-Resolution, Multi-Polarization, Multi-Scene, Multi-Source Dataset and Benchmark for Optical-SAR Fusion Object Detection},
  author={Wang, Chao and Lu, Wei and Li, Xiang and Yang, Jian and Luo, Lei},
  journal={arXiv preprint arXiv:2505.10931},
  year={2025}
}

@article{shen2025multispectral,
  title={Multispectral state-space feature fusion: Bridging shared and cross-parametric interactions for object detection},
  author={Shen, Jifeng and Zhan, Haibo and Dong, Shaohua and Zuo, Xin and Yang, Wankou and Ling, Haibin},
  journal={Information Fusion},
  volume={127},
  part={C},
  pages={103895},
  year={2025},
  publisher={Elsevier}
}

@article{shen2026irdfusion,
  title={IRDFusion: Iterative relation-map difference guided feature fusion for multispectral object detection},
  author={Shen, Jifeng and Zhan, Haibo and Zuo, Xin and Fan, Heng and Yuan, Xiaohui and Li, Jun and Yang, Wankou},
  journal={Pattern Recognition},
  pages={113189},
  year={2026},
  publisher={Elsevier}
}
```
