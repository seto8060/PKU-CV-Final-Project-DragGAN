# DragDiffusion Comparison Experiments

本文件夹包含与 **DragDiffusion** 方法对比实验相关的代码与数据。由于完整实验仓库体量较大，此处仅保留用于实验运行的核心脚本以及对应的数据集，其余内容可通过官方仓库获取。

## 数据集说明

`dataset/` 文件夹中存放用于对比实验的数据集，分为 **In-Domain** 与 **Open-Domain** 两部分：

- **In-Domain**  
  由人工标注的 10 组数据组成，用于评估模型在已知分布内的编辑性能。

- **Open-Domain**  
  选取自 **DRAGBENCH** 数据集的 20 组样本，用于评估模型在开放域场景下的泛化能力。

每组数据均包含以下文件：
- `meta_data.pkl`：包含编辑掩码、控制点等元数据；
- `original_image.png`：原始输入图像；
- `user_drag.png`：用户拖拽示意图（仅用于可视化说明）。

所有实验的完整定量结果已汇总至 `full_result.xlsx` 文件中。

## DragDiffusion 实验环境配置

实验所使用的 **DragDiffusion** 相关代码可从官方仓库获取：  
https://github.com/Yujun-Shi/DragDiffusion

环境配置流程如下：

```bash
conda env create -f environment.yaml
conda activate dragdiff
apt update
apt install -y libsm6 libxext6 libxrender1
pip install -U huggingface_hub==0.25.2
pip install -U gradio==3.50.2
```
完成上述步骤后，需手动下载并放置相应的预训练模型，即可正常运行实验代码。

## PTI 反演设置

为使 DragGAN 能够处理真实图像，实验采用 **Pivotal Tuning Inversion (PTI)** 方法对输入图像进行反演。  
PTI 代码可从以下仓库获取：  
https://github.com/danielroich/PTI

在运行前需进行如下修改与设置：

1. 将 `training/coaches/single_id_coach.py` 替换为 `PTI/single_id_coach.py` 中对应的实现；
2. 在 PTI 根目录下运行 `PTI/run_pti_dataset.py`

## 定量评估

实验结果使用以下三项指标进行评估（具体定义详见实验报告）：

- **Image Fidelity (IF)**
- **Background Preservation (BP)**
- **Mean Distance (MD)**

评估过程依赖 **DIFT** 特征提取工具，其代码可从以下仓库下载：  
https://github.com/Tsingularity/dift

完成依赖配置后，在项目根目录下运行 `evaluation/evaluation.py`

即可获得对应的定量评估结果。

