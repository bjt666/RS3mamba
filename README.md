# RS³Mamba 马尾藻语义分割基线

基于 [RS³Mamba 官方实现](https://github.com/sstary/SSRS/tree/main/RS3Mamba)，适配高分影像马尾藻二分类。代码在本地维护并通过 GitHub 同步，训练由用户在 Linux / AutoDL 服务器运行。

## 模型与实验设置

- 保留原版 RS³Mamba 的 ResNet-18 主分支、VMamba-Tiny 辅助分支、CCM 和解码器。两条分支输入同一张三通道假彩色图，不需要 DSM。
- 固定上游提交 `b89998178ab07679c179fbda3b28eefcd0e361eb`。源码、许可证和修改记录见 [vendor/rs3mamba/PROVENANCE.md](vendor/rs3mamba/PROVENANCE.md)。移除未使用的 MONAI 导入并调整包路径，不修改模型前向计算。
- CNN 分支加载 `swsl_resnet18`，权重文件 `semi_weakly_supervised_resnet18-118f1556.pth`。这是 Meta 的 SWSL 权重，使用过额外预训练数据，实验报告应注明，不能笼统标为仅 ImageNet-1K。
- Mamba 分支加载旧版 VMamba v0 Tiny 的 `vssmtiny_dp01_ckpt_epoch_292.pth`，在项目中命名为 `pretrain/vmamba_tiny_e292.pth`。不要替换为 VMamba v2 / s1l8 权重。
- 严格检查加载的键和形状，记录源权重哈希、加载覆盖情况及跳过项到 `initialization.json`。输入 stem、融合与解码器等未加载部分保留自身初始化。
- 对 CNN 的 RGB 首层采用与现有 SegFormer 相同的 NIR/R/G 适配：NIR 使用 RGB 卷积权重均值，R/G 使用原对应权重。这属于本数据集的输入适配，需随实验报告记录。

本工程是 **RS³Mamba 在 GF 马尾藻数据上的任务适配基线**。数据预处理、优化器、物理 batch 等与原论文存在差异，不能称为原论文全部训练设置的逐项复现。

| 配置 | 优化器与学习率 | 训练预算 | 用途 |
| --- | --- | --- | --- |
| `configs/rs3mamba_gf1.yaml`（默认） | AdamW，编码器 `6e-5`，融合/解码器 `6e-4`，wd `0.01`，余弦衰减，CE | 100 轮；物理 batch=4，累积=2 | 与现有 SegFormer 使用相近的训练配方，作为后续结构消融的固定起点 |
| `configs/rs3mamba_official_sgd.yaml` | SGD，lr `0.01`，momentum `0.9`，wd `0.0005`，第 25/35/45 轮乘 `0.1`，CE | 50 轮；物理 batch=2，累积=5 | 保留官方优化器和调度设置的可选实验 |

梯度累积的名义有效 batch 分别为 8 和 10，最后一组按实际样本数归一化。**累积不能使 BatchNorm 与对应的大物理 batch 等价**。两份配置应作为不同实验分别报告。不要混用其中较好的测试集结果选择配方；根据验证集确定设置。

## 1. 服务器与环境

建议 Ubuntu 20.04 / 22.04，Python 3.10，NVIDIA GPU。优先选择 24 GB 显存；已有 RTX 3080 Ti 12 GB 可先按默认 batch=4 试跑，实际显存以服务器检查为准。默认 FP32；需要时在独立实验中启用 AMP 并记录。

固定使用 PyTorch 2.1.2、torchvision 0.16.2、CUDA 11.8、CUDA Toolkit 11.8（含 `nvcc`）及 GCC/G++。CUDA 11.8 开发镜像配合 GCC/G++ 11 是建议起点。`nvidia-smi` 显示的 CUDA 版本是驱动支持上限，编译版本以 `nvcc --version` 为准。

即使复用之前的 AutoDL 实例，也建议使用独立 Conda 环境。为随系统盘镜像保存，环境明确放在 `/root/rs3mamba_env`，使用 `--prefix` 绕过可能指向数据盘的 `envs_dirs` 设置，并用 `--copy` 安装 Conda 包。下面不会改动 SegFormer 的 Python 环境。不需要安装 MMCV、MMSegmentation、MONAI 或 Apex。

```bash
cd /root
git clone https://github.com/bjt666/RS3mamba.git
cd /root/RS3mamba

nvidia-smi
nvcc --version
g++ --version

source /root/miniconda3/etc/profile.d/conda.sh
# 首次创建；已有此目录时跳过创建，避免覆盖。
if [ ! -e /root/rs3mamba_env ]; then
  conda create --prefix /root/rs3mamba_env python=3.10 --copy -y
fi
conda activate /root/rs3mamba_env
python -c "import sys; assert sys.prefix == '/root/rs3mamba_env', sys.prefix; print(sys.executable)"
python -m pip install --no-cache-dir pip==24.3.1 setuptools==69.5.1 wheel==0.43.0
python -m pip install --no-cache-dir torch==2.1.2 torchvision==0.16.2 --index-url https://download.pytorch.org/whl/cu118
python -m pip install --no-cache-dir -r requirements.txt
MAX_JOBS=4 python -m pip install --no-cache-dir causal-conv1d==1.1.1 --no-build-isolation --no-deps
MAX_JOBS=4 python -m pip install --no-cache-dir mamba-ssm==1.1.1 --no-build-isolation --no-deps
python -m pip check
```

源码编译可能需要数分钟。若 `nvcc` 不存在或版本不同，先修正 Toolkit，不要直接升级到最新 Mamba、PyTorch 或 Triton 来绕过报错。CUDA 扩展版本另存于 `requirements-cuda.txt`。

GitHub 连接超时时，可在支持此功能的 AutoDL 实例终端执行 `source /etc/network_turbo` 后重试。如果实例没有该文件，按平台网络说明配置。首次权重下载需要网络；训练、续训、评估和预测均使用本地文件，不在内部自动访问 Hugging Face。

## 2. 下载两套预训练权重并检查环境

所有命令在 `/root/RS3mamba` 下运行：

```bash
python tools/download_pretrained.py
python tools/download_pretrained.py --verify-only
python tools/check_environment.py --model-smoke --output outputs/environment_check.json
```

下载工具仅从 Meta 和 VMamba 官方公开地址获取文件，校验已固定的文件大小及哈希，保留下载记录。下载目录为：

```text
pretrain/
  semi_weakly_supervised_resnet18-118f1556.pth
  vmamba_tiny_e292.pth
```

也可在本地运行下载工具，再自行上传整个 `pretrain/` 目录。`--output-dir` 可指定下载位置；训练使用其他位置时传 `--cnn-checkpoint` 和 `--vmamba-checkpoint`。从项目根目录启动后，相对路径按当前目录解析。

环境检查先验证实际 selective-scan CUDA 前向/反向，再以 batch=2、256×256、FP32 检查完整模型和预训练加载。出现错误时保留完整终端输出及 JSON；检查通过后再开始正式训练。只收集版本而不做计算可用：

```bash
python tools/check_environment.py --report-only --output outputs/environment_report.json
```

## 3. 自行放入数据集

GitHub 不包含数据、权重或训练结果。将自己的数据放到以下位置，或用 `--data-root` 指向实际目录：

```text
dataset/
  images/
    train/*.tif
    val/*.tif
    test/*.tif
  masks/
    train/*.tif
    val/*.tif
    test/*.tif
```

图像和 mask 文件名需完全相同，支持 `.tif` / `.tiff`。本基线要求 HWC **256×256×4** 图像，波段依次为 **蓝、绿、红、近红外**；单通道 mask 尺寸为 256×256，允许 0/1/255，其中 0 为背景，1 或 255 为马尾藻，255 不作为忽略标签。

读取原始浮点 TIFF 并转为 float32，选择 `[3,2,1]`（零起始，即 NIR/R/G）。不除以 255，不做 RGB/BGR 交换，不将 TIFF 转 JPG。归一化均值/标准差仅从 train 计算，并保存到完整检查点；验证、测试和预测复用同一统计。训练增强只有同步水平翻转、垂直翻转和 90° 旋转，无颜色扰动或随机缩放。

```bash
python prepare.py --data-root dataset --output-dir outputs/data_check
```

先查看 `outputs/data_check/stats.json` 和 `previews/`，确认波段及标签对齐。预览中的 2%–98% 拉伸仅用于显示。检查会报告完全相同的选定通道输入；跨划分重复会失败。内容哈希不能排除来自同一场景的相邻或重叠切片泄漏，需要按原始场景/空间位置保证划分合理。

已知十张全零训练影像保留并列入报告，不自动删除。训练入口不访问 test；按验证集前景 IoU 选择最佳权重。

## 4. 短程试跑

```bash
python train.py \
  --config configs/rs3mamba_gf1.yaml \
  --data-root dataset \
  --output-dir outputs/rs3mamba_smoke_s42 \
  --epochs 1 --max-train-batches 3 --max-val-batches 2 \
  --num-workers 0
```

必须完成真实模型前向、反向、验证和 checkpoint 保存。这里的指标只用于检查流程，不能作为正式结果。正式训练另起目录、重新从预训练初始化，不续接 smoke 权重。

如果显存不足，先在新目录加 `--amp` 重试；或把物理 batch 调整为 `--batch-size 2 --accumulation-steps 4`，保持名义有效 batch=8，并在所有结构消融中固定相同设置。AMP 若产生非有限 loss，应回到 FP32 排查，不接续异常实验。

## 5. 正式训练与断点恢复

优先运行默认 100 轮配置：

```bash
python train.py \
  --config configs/rs3mamba_gf1.yaml \
  --data-root dataset \
  --output-dir outputs/rs3mamba_gf1_s42
```

为了在 SSH 断开后继续，可以使用 tmux（若镜像未安装，可安装系统包）：

```bash
tmux new -s rs3mamba
conda activate /root/rs3mamba_env
cd /root/RS3mamba
python -u train.py --config configs/rs3mamba_gf1.yaml --data-root dataset --output-dir outputs/rs3mamba_gf1_s42
```

按 `Ctrl+B` 后按 `D` 脱离会话；重新进入使用 `tmux attach -t rs3mamba`。不要同时在同一实验目录启动两个进程。

恢复最近一个完整 epoch（当前中断的 epoch 会重新运行）：

```bash
python train.py \
  --resume outputs/rs3mamba_gf1_s42/checkpoints/last.pt \
  --data-root dataset
```

恢复时自动还原配置、模型、优化器、学习率调度器、AMP scaler、随机数和 DataLoader generator。数据指纹必须一致。训练总轮数、batch、累积次数、种子、AMP、学习率等不能在续训中更改；换设置请创建新实验。移动服务器时复制整个实验目录。续训不需要初始预训练文件。

如果需要运行另一份 SGD 方案，创建独立实验：

```bash
python train.py --config configs/rs3mamba_official_sgd.yaml --data-root dataset --output-dir outputs/rs3mamba_sgd_s42
```

若做从零训练消融，使用独立目录并添加 `--no-pretrained`，它同时关闭两条分支的预训练。

## 6. 评估测试集、保存预测图

先根据验证曲线确认配置和训练预算，再评估 test。评估读取训练检查点中的模型及预处理，不额外传训练 YAML：

```bash
python evaluate.py \
  --checkpoint outputs/rs3mamba_gf1_s42/checkpoints/best.pt \
  --data-root dataset --split test
```

结果默认写到该实验的 `evaluation/test-时间戳/`：

- `metrics.json`：前景 IoU、Dice/F1、Precision、Recall、mIoU、OA，以及混淆矩阵。
- `per_image_metrics.csv`：逐图指标。
- `masks/`：全部图片的 0/1 类别 PNG，正常看图时可能近乎全黑。
- `visualizations/`：选定样本的假彩色、GT、预测和对比图。

指标均为 0–1 比例，乘以 100 为百分数。以全数据集累计的像素混淆矩阵计算指标，不把逐图 IoU 直接平均。分母为零时写 `null`，与现有 SegFormer 的指标策略一致。

无需标签的图像预测：

```bash
python predict.py \
  --checkpoint outputs/rs3mamba_gf1_s42/checkpoints/best.pt \
  --input dataset/images/test \
  --output-dir outputs/rs3mamba_gf1_s42/predictions/test
```

`--input` 可为一张 TIFF 或包含 TIFF 的目录，同样要求 256×256×4。原始 TIFF 不会被覆盖。

## 7. 实验记录与后续改进

每个实验保存 `config.yaml`、`preprocessing.json`、`initialization.json`、`model_summary.json`、`environment.json`、`requirements-freeze.txt`、数据指纹、日志、`metrics.csv`、曲线、`summary.json` 和 `checkpoints/best.pt` / `last.pt`。最优模型由验证集马尾藻 IoU 决定。

原版、模块 A、模块 B、A+B 应分别从相同初始预训练文件独立训练，固定数据划分、输入适配、训练配方及预算；新增模块使用自身初始化。重要结果补充多个随机种子。SWSL 与仅 ImageNet 预训练来源有差异，比较 SegFormer 时需明确记录。

源代码更新使用 `git pull --ff-only origin main`。当前实验应记录其开始时的 Git 提交；更新代码后优先新建实验目录，避免混淆已完成的结果。数据、权重、输出、临时测试依赖均由 `.gitignore` 排除。

## 8. 连同 AutoDL 镜像保存

AutoDL 的「保存镜像」保存实例的系统盘；`/root/autodl-tmp` 数据盘和 `/root/autodl-fs` 等独立挂载不包含在系统盘镜像中。环境、源码、需要一并保留的数据及权重应实际位于系统盘：

```text
/root/rs3mamba_env/          # Python、PyTorch、已编译 Mamba 扩展
/root/RS3mamba/             # 源码
/root/RS3mamba/dataset/     # 自行上传的实际数据文件
/root/RS3mamba/pretrain/    # 下载的初始权重
/root/RS3mamba/outputs/     # 已保存的日志和训练 checkpoint
```

如果上述路径是指向数据盘的软链接，链接目标里的文件仍不会进入镜像。上传前后可通过 `readlink -f` 和 `df -h` 核对实际位置；系统盘空间不足时先在平台扩容系统盘，不将需要进镜像的环境迁到数据盘。

在环境检查和试跑通过后，保存环境清单：

```bash
conda activate /root/rs3mamba_env
cd /root/RS3mamba
mkdir -p environment
python -m pip freeze > environment/pip-freeze.txt
conda env export --prefix /root/rs3mamba_env > environment/conda-environment.yml
git rev-parse HEAD > environment/code-commit.txt
python tools/check_environment.py --report-only --output environment/runtime.json
readlink -f /root/rs3mamba_env /root/RS3mamba
df -h /root /root/rs3mamba_env /root/RS3mamba
sync
```

完成训练或确认 checkpoint 已落盘后，在 AutoDL 控制台**关机 → 更多操作 → 保存镜像**，等保存完成后在「我的镜像」确认。环境本身会随系统盘保存，不是仅依靠上述清单重装；清单用于记录版本。Git 忽略规则只影响推送 GitHub，不影响系统盘镜像。

用此镜像创建实例后，在终端重新激活环境即可：

```bash
source /root/miniconda3/etc/profile.d/conda.sh
conda activate /root/rs3mamba_env
cd /root/RS3mamba
python tools/check_environment.py
```

镜像保留文件和已安装扩展，不保留关机前仍在运行的进程。训练从 `checkpoints/last.pt` 恢复。平台规则参考 [系统盘/数据盘说明](https://www.autodl.com/docs/env/) 和 [镜像说明](https://www.autodl.com/docs/image/)。

## 本地验证及限制

本机 Windows 用于数据、模型加载和训练控制逻辑检查；正式运行环境为上面指定的 Linux CUDA 环境。真实 selective-scan CUDA 算子、服务器显存占用和收敛效果必须由服务器的环境检查和短程训练验证。项目不会用替代计算冒充真实 Mamba GPU 试跑。

2026-10-26 本地检查：20 项自动测试通过，覆盖数据与标签、归一化与指标、预训练加载、梯度累积、checkpoint/RNG 恢复，以及使用测试模型的训练→中断→续训→评估→预测流程。两套实际官方权重通过完整 SHA-256 校验，模型构建时 CNN 的 100/100、VMamba 的 205/205 个适用张量加载成功。模型加载测试中的 selective-scan 导入仅为测试隔离，禁止执行替代扫描；没有据此宣称完整模型的 CUDA 前向/反向通过。

本地检查使用 Windows Python 3.10 / PyTorch 2.5.1；目标服务器固定版本为 Linux Python 3.10 / PyTorch 2.1.2+cu118，后者尚未在本机执行。数据检查确认 700/87/89 对、训练集前景约 4.84%，未发现跨划分完全相同的选定通道输入。没有将任何数据集或本地检查输出提交到 Git。

本地自动检查命令（已安装 PyTorch 和相关基础依赖时）：

```bash
python -m unittest discover -s tests -v
```

## 引用与来源

- [RS³Mamba 论文](https://doi.org/10.1109/LGRS.2024.3414293)
- [官方 RS³Mamba](https://github.com/sstary/SSRS/tree/main/RS3Mamba)
- [VMamba v0 分类权重](https://github.com/MzeroMiko/VMamba/releases/tag/%23v0cls)
- [Meta SWSL ResNet 权重](https://github.com/facebookresearch/semi-supervised-ImageNet1K-models)
- [PyTorch 历史版本安装命令](https://pytorch.org/get-started/previous-versions/)

上游源码遵循所附 Apache-2.0 许可证。独立下载的 SWSL 权重另有其来源许可，未放入 Git 仓库。
