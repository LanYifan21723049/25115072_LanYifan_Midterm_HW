# 实验结果说明

本目录包含所有实验的训练结果、模型检查点和可视化图表。

## ? 目录结构

```
results/
├── figures/                          # 训练曲线图和可视化
│   ├── ptb_training_results.png     # PTB语言模型训练曲线
│   ├── main_training_loss.png       # 翻译模型训练损失
│   ├── main_training_perplexity.png # 翻译模型困惑度
│   ├── ablation_size_full_15epochs.png      # 完整消融对比
│   ├── ablation_size_full_curves.png        # 消融训练曲线
│   └── ablation_parameter_efficiency.png    # 参数效率分析
├── ptb_model_fixed/                 # PTB语言模型结果
│   ├── config.json                  # 模型配置
│   ├── training_history.json        # 训练历史
│   └── checkpoints/                 # 模型检查点
│       └── best_model.pt
├── translation/                      # 基准翻译模型结果
│   ├── config.json
│   ├── training_history.json
│   ├── evaluation_results.json
│   └── checkpoints/
│       └── best_model.pt
├── ablation_quick/                   # 快速消融实验结果（5K数据）
│   ├── ablation_results.json
│   ├── base_config.json
│   ├── ablation_heads/              # 注意力头数消融
│   ├── ablation_layers/             # 层数消融
│   ├── ablation_dropout/            # Dropout消融
│   └── ablation_size/               # 模型大小消融
└── ablation_full/                    # 完整消融实验结果（200K数据）
    ├── ablation_results.json
    ├── base_config.json
    └── ablation_size/               # 模型大小完整消融
        ├── d256_h8_l3_dr0.1/        # 基准: 11.7M参数
        ├── d512_h8_l3_dr0.1/        # 中等: 46.5M参数
        └── d768_h12_l3_dr0.1/       # 大型: 104M参数
```

## ? 主要实验结果

### 1. PTB语言建模（Encoder-Only）

**配置**：
- 架构：Encoder-Only（4层）
- 参数量：约15M
- 训练数据：Penn Treebank
- 训练时长：约2-3小时（RTX 3090）

**最佳结果**：
- 验证Loss：3.67
- 测试Loss：3.71
- 测试Perplexity：40.8
- Token准确率：48.7%

**可视化**：`figures/ptb_training_results.png`

---

### 2. Europarl机器翻译（Encoder-Decoder）

**配置**：
- 架构：Encoder-Decoder（3层×2）
- 参数量：约11.7M
- 训练数据：Europarl v7（200K句对）
- 训练时长：约3-4小时（RTX 3090）

**最佳结果**：
- 验证Loss：1.93
- 测试Loss：1.98
- 测试Perplexity：6.88
- BLEU分数：23.7

**可视化**：
- `figures/main_training_loss.png` - 训练损失曲线
- `figures/main_training_perplexity.png` - 困惑度曲线

---

### 3. 快速消融实验（5K数据，5 epochs）

**目的**：快速测试多个超参数的影响

**测试维度**：
1. 注意力头数：1, 2, 4, 8
2. 模型层数：1, 2, 3
3. Dropout率：0.0, 0.1, 0.2, 0.3
4. 模型大小：d_model=256, 512

**关键发现**：
- d_model=512 比 d_model=256 提升6.6%
- 8个注意力头最优
- Dropout=0.1最佳
- 3层深度平衡性能和效率

**总耗时**：约4小时（16个实验配置）

---

### 4. 完整消融实验（200K数据，15 epochs）

**目的**：在完整数据集上验证模型规模影响

**测试配置**：

| 模型 | d_model | 参数量 | 验证Loss | vs基准改善 | 训练时间 |
|------|---------|--------|----------|-----------|---------|
| Small | 256 | 11.7M | 2.002 | - (基准) | ~2.9h |
| Medium | 512 | 46.5M | **1.888** | **+5.7%** | ~2.9h |
| Large | 768 | 104M | **1.873** | **+6.4%** | ~2.9h |

**关键发现**：
- 模型规模显著影响性能
- 参数效率呈边际递减
- Medium模型（d=512）性价比最优
- 15 epochs的大模型超越30 epochs的小模型

**可视化**：
- `figures/ablation_size_full_15epochs.png` - 性能对比柱状图
- `figures/ablation_size_full_curves.png` - 训练曲线对比
- `figures/ablation_parameter_efficiency.png` - 参数效率散点图

**总耗时**：约8.7小时（3个模型）

---

## ? 图表说明

### ptb_training_results.png
展示PTB语言模型的训练过程：
- 上图：训练和验证损失随epoch变化
- 下图：训练和验证困惑度随epoch变化
- 显示模型在30个epoch内稳定收敛

### main_training_loss.png
展示Europarl翻译模型的损失曲线：
- 蓝线：训练损失
- 红线：验证损失
- 红星标记：最佳验证损失点（Val Loss=1.93）

### main_training_perplexity.png
展示翻译模型的困惑度变化：
- 困惑度从初始的约130降至最终的6.88
- 验证困惑度在20 epoch后趋于稳定

### ablation_size_full_15epochs.png
对比三种模型规模的性能：
- 柱状图展示三个模型的验证损失
- 橙色虚线表示30 epoch基准模型（Val Loss=1.93）
- 标注显示相对基准的改善百分比

### ablation_size_full_curves.png
展示三个模型的训练曲线：
- 蓝色：Small (d=256)
- 红色：Medium (d=512)
- 绿色：Large (d=768)
- 显示更大模型收敛更快且性能更好

### ablation_parameter_efficiency.png
参数效率散点图：
- X轴：模型参数量（百万）
- Y轴：相对基准的改善百分比
- 展示参数增加带来的性能提升趋势

---

## ? 如何重现这些结果

所有实验都可以通过以下命令重现（含固定随机种子seed=42）：

```bash
# 1. PTB语言模型
bash scripts/run.sh ptb

# 2. Europarl翻译模型
bash scripts/run.sh translation

# 3. 快速消融实验
bash scripts/run.sh quick_ablation

# 4. 完整消融实验
bash scripts/run.sh full_ablation

# 5. 生成可视化
bash scripts/run.sh visualize

# 运行所有实验
bash scripts/run.sh all
```

详细的命令行参数请参考项目根目录的 `README.md`。

---

## ? 数据文件说明

### config.json
包含模型的完整配置信息：
- 模型超参数（d_model, n_heads, d_ff等）
- 训练超参数（lr, batch_size, epochs等）
- 数据集信息
- 随机种子

### training_history.json
记录每个epoch的训练历史：
```json
{
    "train_losses": [4.79, 3.20, 2.76, ...],
    "val_losses": [3.42, 2.72, 2.46, ...]
}
```

### evaluation_results.json
模型在测试集上的评估结果：
- Test Loss
- Test Perplexity
- BLEU分数（翻译任务）
- 其他指标

### checkpoints/best_model.pt
包含最佳模型的完整状态：
- model_state_dict：模型参数
- optimizer_state_dict：优化器状态
- epoch：训练轮数
- best_val_loss：最佳验证损失

---

## ?? 加载模型示例

```python
import torch
from src.transformer import Transformer

# 加载配置
import json
with open('results/translation/config.json', 'r') as f:
    config = json.load(f)

# 创建模型
model = Transformer(**config)

# 加载权重
checkpoint = torch.load('results/translation/checkpoints/best_model.pt')
model.load_state_dict(checkpoint['model_state_dict'])

# 推理模式
model.eval()
```

---

## ? 更多信息

详细的实验设计、理论推导和结果分析请参考：
- LaTeX报告：`report/report.pdf`
- 项目README：`README.md`
- 消融实验指南：`过程说明文件/QUICK_ABLATION_GUIDE.md`
