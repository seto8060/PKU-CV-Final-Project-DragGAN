# DragGAN 背景保持优化研究

本项目提供了一个基于 GragGAN 的背景保持能力增强的可视化交互界面，支持用户通过交互式操作进行图像编辑，并可精细控制背景保持程度。此外，项目允许用户自定义创建、保存实验数据集。基于已创建的数据集，项目支持参数化实验配置，可调节优化参数，自动保存中间实验图像及实验数据。
## 环境要求

若您拥有支持 CUDA 的显卡，请遵循 [NVlabs/stylegan3](https://github.com/NVlabs/stylegan3#requirements) 的官方要求。

通常的安装步骤如下，这些命令将设置正确的 CUDA 版本并安装所有 Python 依赖包：  

```
conda env create -f environment.yml
conda activate stylegan3
pip install -r requirements.txt
```

若不使用 NVIDIA CUDA（例如在 macOS Silicon M1/M2 上使用 GPU 加速，或仅使用 CPU），请尝试以下步骤：

```sh
cat environment.yml | \
  grep -v -E 'nvidia|cuda' > environment-no-nvidia.yml && \
    conda env create -f environment-no-nvidia.yml
conda activate stylegan3

# On MacOS
export PYTORCH_ENABLE_MPS_FALLBACK=1
```

## 在 Docker 中运行 Gradio 可视化工具

提供的 Docker 镜像基于 NGC PyTorch 仓库构建。若需在 Docker 容器中快速体验可视化工具，请运行以下命令：

```sh
docker build . -t draggan:latest  

docker run -p 7860:7860 -v "$PWD":/workspace/src -it draggan:latest bash

cd src && python visualizer_drag_gradio.py --listen
```

## 下载预训练 StyleGAN2 权重

```
python scripts/download_model.py
```


## 使用方法
### 运行 DragGAN Gradio 交互界面
```bash
python visualizer_drag_gradio.py
```
在可视化界面可以通过调整 Blend Interval (N) 和 Reproject Steps (M) 参数，来控制优化过程中特征融合的频率与强度

### 基于已有数据的参数化实验
```bash
# 对单个实验数据进行实验
python run_experiment.py -e experiment_stylegan2-ffhq-512x512_seed7_421929.json -b 50 -r 0 ./experiment_results/50-0

# 对指定目录下所有实验数据进行实验
python run_experiment.py -b 50 -r 50 ./experiment_results/50-50
```
运行实验脚本时，可配置以下参数：
1. `--experiment-data-dir / -d` 实验数据目录
   
   该目录存放实验所需的 JSON 配置文件，默认值：./experiment_data

2. `--experiment / -e` 指定实验数据文件
   
   默认值：运行实验数据目录中的所有实验

3. `--steps / -s` 总优化步数
   
   默认值：201

4. `--save-interval / -i` 
   
   每隔 N 步保存一次未标注图像和标注图像，默认值：50

5. `--blend-interval / -b` 设置特征融合操作的频率
   
   默认值：50 步执行一次融合

6. `--reproject-steps / -r` 控制每次反演优化的步数
   
   默认值：25

7. 位置参数 `output_base_dir` 实验结果的输出目录
   
   命名通常为 ./experiment_results/{blend_interval}-{reproject_steps}
   
每次执行特征融合与重投影（GAN 反演）阶段时，都会生成三张对应的可视化图像：`before blend`, `after blend`, and `after reprojection`。这些图像将保存在 `blend_visualization` 文件夹中，同时标注图像和未标注图像也会保存在指定的输出目录中。

### 创建自定义实验数据
```bash
# 请确保已运行过 DragGAN Gradio 交互界面
python create_data.py
```
您可以在可视化界面中自定义操作点和可编辑区域，并将其保存为一组实验数据。每组数据将包含原始图像、操作点位置以及掩码区域的信息。

创建的数据将保存在 `experiment_data` 文件夹中，可供后续实验使用。

## 致谢

我们的代码建立在 [DragGAN](https://github.com/XingangPan/DragGAN/) 基础上。

