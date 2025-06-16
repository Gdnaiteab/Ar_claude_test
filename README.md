# AR预测视频模型项目

基于Transformer的自回归气象数据预测模型，用于预测未来时刻的气象场数据。

## 📋 项目概述

本项目实现了一个基于Vision Transformer (ViT) 的自回归预测模型，能够通过过去若干时刻的多变量气象数据，预测未来时刻的气象场分布。模型采用patch-based的token化方式，结合时空区块划分和自回归掩码机制，实现高精度的气象数据预测。

### 🎯 主要特性

- **多变量支持**: 支持69个气象变量（单层和多层变量）
- **时空建模**: 采用ViT patch embedding + Transformer架构
- **自回归预测**: 支持单步和多步预测
- **灵活配置**: 基于YAML的完整配置系统
- **实验跟踪**: 集成Wandb进行训练监控和可视化
- **数据完整性**: 自动检测和插值缺失数据
- **可视化**: 内置预测结果可视化功能

## 📁 项目结构

```
ar_claude_test/
├── checkpoints/          # 模型检查点存储
├── config/              # 配置文件
│   └── ar_pred_video_vanilla.yaml
├── log/                 # 训练日志
├── models/              # 模型架构
│   ├── __init__.py
│   └── ar_pred_video_vanilla.py
├── tmp/                 # 临时文件
├── utils/               # 工具函数
│   ├── __init__.py
│   ├── config.py
│   ├── dataloader.py
│   └── logger.py
├── wandb/               # Wandb数据
├── train_ar_pred_video_vanilla.py    # 训练脚本
├── inference_ar_pred_video_vanilla.py # 推理脚本
├── setup_env.sh         # 环境设置脚本
├── requirements.txt     # 依赖包
├── .gitignore
└── README.md
```

## 🚀 快速开始

### 1. 环境设置

```bash
# 创建conda环境
./setup_env.sh

# 激活环境
conda activate ar_pred

# 安装依赖
pip install -r requirements.txt
```

### 2. 数据准备

确保您的数据按以下结构组织：

```
/sharefiles4/qubohuan/
├── t2m_npy/             # 2米温度（单层）
│   ├── 1979/
│   │   ├── 1979-0000.npy
│   │   ├── 1979-0001.npy
│   │   └── ...
│   ├── mean_std.json
│   └── min_max.json
├── r_npy/               # 相对湿度（13层）
│   ├── level_00/
│   │   ├── 1979/
│   │   ├── mean_std.json
│   │   └── ...
│   └── ...
└── ...
```

**数据格式要求:**
- 每个`.npy`文件形状: `(128, 256)`, 类型: `float32`
- 数据已经标准化处理
- `mean_std.json`包含均值和标准差信息

### 3. 配置文件

编辑 `config/ar_pred_video_vanilla.yaml` 来调整模型和训练参数：

```yaml
# 数据配置
data:
  data_root: "/sharefiles4/qubohuan/"  # 修改为您的数据路径
  time_interval: 6                     # 时间间隔（小时）
  variables:                          # 使用的变量配置
    single_level: ["t2m_npy", "tp_npy", "u10_npy", "v10_npy"]
    multi_level:
      - variable: "r_npy"
        levels: ["level_00"]

# 模型配置
model:
  time_window: 8                      # 时间窗口大小
  patch_size: [16, 16]               # patch大小
  embed_dim: 768                     # 嵌入维度
  
# 训练配置
training:
  num_epochs: 100
  learning_rate: 1e-4
  batch_size: 16
```

### 4. 训练模型

```bash
# 开始训练
python train_ar_pred_video_vanilla.py --config config/ar_pred_video_vanilla.yaml

# 从检查点恢复训练
python train_ar_pred_video_vanilla.py --config config/ar_pred_video_vanilla.yaml --resume checkpoints/latest_model.pth
```

### 5. 模型推理

```bash
# 单步预测
python inference_ar_pred_video_vanilla.py \
    --config config/ar_pred_video_vanilla.yaml \
    --checkpoint checkpoints/best_model.pth \
    --input_files file1.npy file2.npy ... \
    --output_dir tmp/inference_results \
    --visualize

# 多步预测
python inference_ar_pred_video_vanilla.py \
    --config config/ar_pred_video_vanilla.yaml \
    --checkpoint checkpoints/best_model.pth \
    --input_files file1.npy file2.npy ... \
    --output_dir tmp/inference_results \
    --multi_step 5 \
    --use_true_data \
    --visualize
```

## 🏗️ 模型架构

### AR预测视频Vanilla模型

模型采用以下架构设计：

1. **Patch Embedding**: 将输入的时空数据转换为token序列
   - 输入: `(B, T, H, W, V)` → 重塑为 `(B, T*V, H, W)`
   - Patch化: `(B, num_patches*T, embed_dim)`

2. **位置编码**: 为每个patch添加可学习的位置编码

3. **时空区块划分**: 将tokens按时间维度划分为T个区块

4. **Random Patchify**: 随机移除一个区块作为预测目标

5. **Transformer Encoder** (8层):
   - 多头自注意力（带自回归掩码）
   - MLP + 残差连接
   - LayerNorm

6. **Transformer Decoder** (4层):
   - 交叉注意力（无自注意力）
   - 使用缺失区块的位置编码引导生成

7. **输出头**: 重建为原始样本格式 `(B, H, W, V)`

### 关键特性

- **自回归掩码**: 按掩码率随机掩盖后续区块的注意力
- **时空建模**: 同时处理时间和空间维度
- **多变量融合**: 支持单层和多层气象变量
- **数据完整性**: 自动处理缺失数据插值

## 📊 训练监控

项目集成了Wandb进行实验跟踪：

- **损失曲线**: 训练/验证/测试损失
- **学习率调度**: 实时监控学习率变化
- **可视化样本**: 定期保存预测vs真实值对比图
- **模型性能**: 多种评估指标跟踪

## 🔧 配置说明

### 数据配置

- `data_root`: 数据根目录路径
- `time_interval`: 采样时间间隔（1=每小时，6=每6小时）
- `variables`: 使用的变量配置（最少9个变量）
- `years`: 训练/验证/测试年份划分

### 模型配置

- `time_window`: 时间窗口大小（默认8）
- `patch_size`: patch尺寸（默认16x16）
- `embed_dim`: 嵌入维度（默认768）
- `random_patchify`: 是否随机选择缺失区块
- `mask_ratio`: 自回归掩码比例

### 训练配置

- `num_epochs`: 训练轮数
- `learning_rate`: 学习率
- `optimizer`: 优化器类型（adam/adamw/sgd）
- `lr_scheduler`: 学习率调度策略
- `early_stopping`: 早停配置

## 📈 性能优化

### 建议配置

**GPU内存优化:**
- 调整`batch_size`根据GPU内存
- 启用`mixed_precision`训练
- 使用`num_workers`并行数据加载

**训练效率:**
- 使用余弦学习率调度
- 设置合理的`eval_every`频率
- 启用检查点保存策略

**模型调优:**
- 调整`mask_ratio`影响训练难度
- 修改`time_window`平衡精度和效率
- 优化`patch_size`适应数据分辨率

## 🐛 常见问题

### Q: 数据加载失败
**A**: 检查数据路径、文件格式和mean_std.json文件是否存在

### Q: GPU内存不足
**A**: 减少batch_size、启用mixed_precision或使用gradient_checkpointing

### Q: 训练损失不收敛
**A**: 调整学习率、检查数据标准化、增加warmup epochs

### Q: 预测结果异常
**A**: 检查模型输入格式、确认归一化/反归一化正确

## 🤝 贡献指南

1. Fork项目
2. 创建特性分支 (`git checkout -b feature/AmazingFeature`)
3. 提交更改 (`git commit -m 'Add some AmazingFeature'`)
4. 推送分支 (`git push origin feature/AmazingFeature`)
5. 开启Pull Request

## 📄 许可证

本项目采用MIT许可证 - 查看 [LICENSE](LICENSE) 文件了解详情

## 📞 联系方式

如有问题或建议，请通过以下方式联系：

- 项目Issues: [GitHub Issues](https://github.com/Gdnaiteab/Ar_claude_test/issues)
- 邮箱: [您的邮箱]

## 🙏 致谢

感谢以下项目和论文的启发：

- Vision Transformer (ViT)
- Transformer架构
- 气象数据处理相关研究

---

**注意**: 本项目仅用于研究和学习目的，请确保在使用前理解模型的局限性和适用范围。