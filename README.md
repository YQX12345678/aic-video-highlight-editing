# AIC Video Highlight Editing

**2026 AIC 通用视频高光剪辑**: 基于 PyTorch 的视频时序空间裁剪（Auto-Cropping）方案。

## 1. 核心架构
本方案使用 CNN + 时序 Transformer 进行视频帧级别的裁剪框预测：
- **特征提取 (`src/model.py`)**: 采用预训练的 ResNet18 提取每帧的空间特征。
- **时序建模**: 3 层 Transformer Encoder 处理连续 30 帧的特征，理解高光动作的趋势。
- **输出**: 预测每帧的归一化裁剪框 `[cx, cy, w, h]`。

## 2. 损失函数 (`src/loss.py`)
采用 `L1 Loss` + `Temporal IoU Loss` 的混合监督，确保边界框不仅位置准，而且时序上不抖动。

## 3. 真实结果验证 (`outputs/`)
- `outputs/submission_fushi_v7D.jsonl`: 模型真实生成的逐帧预测轨迹（符合官方格式）。
