# Transformer从零实现：完整的Encoder-Only与Encoder-Decoder架构

本项目从零开始实现了完整的Transformer模型，包含**Encoder-Only**架构（用于语言建模）和**Encoder-Decoder**架构（用于机器翻译），并在经典数据集上进行训练、评估和系统的消融实验。这是M502082B《大模型基础与应用》课程的期中作业。

## ? 项目特点

- ?? **完整实现**：从零实现所有核心组件（Multi-Head Attention、FFN、LayerNorm、位置编码等）
- ? **双架构支持**：Encoder-Only（PTB语言模型）+ Encoder-Decoder（Europarl翻译）
- ? **详细文档**：20+页LaTeX报告，包含完整数学推导和训练算法伪代码
- ? **系统消融实验**：两阶段消融（快速5K+完整200K），测试模型大小、头数、层数、dropout
- ? **可视化工具**：训练曲线、注意力热力图、参数效率分析
- ? **可复现**：固定随机种子、详细运行命令、完整超参数记录

## ? 项目结构

```
LMHM1/
├── src/                              # 核心模型实现
│   ├── attention.py                  # 多头自注意力机制
│   ├── feedforward.py                # 逐位置前馈网络
│   ├── positional_encoding.py        # 正弦位置编码
│   ├── encoder.py                    # Transformer Encoder
│   ├── decoder.py                    # Transformer Decoder
│   ├── transformer.py                # 完整Transformer模型
│   ├── data_utils.py                 # 数据加载与预处理
│   └── evaluate.py                   # 模型评估工具
├── scripts/                          # 实验脚本
│   ├── download_ptb.py              # 下载PTB数据集
│   ├── download_translation_data.py  # 下载Europarl数据
│   ├── train_ptb_fixed.py            # PTB语言模型训练
│   ├── train_translation.py          # 翻译模型训练
│   ├── run_quick_ablation.py         # 快速消融（5K，5 epochs）
│   ├── run_ablation_on_full_data.py  # 完整消融（200K，15 epochs）
│   └── generate_visualizations.py    # 生成报告图表
├── data/
│   ├── ptb/                         # Penn Treebank数据
│   └── iwslt2017/                   # Europarl翻译数据（200K句对）
├── results/
│   ├── ptb_model_fixed/             # PTB模型结果
│   ├── translation/                  # 翻译基准模型（Val Loss 1.93）
│   ├── ablation_quick/               # 快速消融结果
│   └── ablation_full/                # 完整消融结果（3个模型规模）
├── report/
│   ├── report.tex                   # LaTeX源文件
│   └── figures/                     # 报告图表
├── requirements.txt                  # Python依赖
└── README.md                         # 本文件
```

## ? 快速开始

### 1. 硬件要求

**推荐配置**：
- **GPU**: NVIDIA RTX 3090 (24GB显存) 或同等性能
- **CUDA**: 11.8+
- **内存**: 32GB RAM
- **磁盘**: 10GB可用空间
- **操作系统**: Linux/Windows 10+

**最低配置**：
- **GPU**: NVIDIA GTX 1080 (8GB显存)
- **内存**: 16GB RAM

> ?? **注意**：本项目所有实验均在GPU上进行。CPU训练未经测试且速度极慢（预计10-20倍时间），不推荐。

### 2. 环境配置

```bash
# 克隆项目
git clone https://github.com/yourusername/LMHM1.git
cd LMHM1

# 创建虚拟环境（推荐）
conda create -n transformer python=3.8
conda activate transformer

# 安装PyTorch（根据CUDA版本选择）
# CUDA 11.8
pip install torch torchvision torchaudio --index-url https://download.pytorch.org/whl/cu118

# 安装其他依赖
pip install -r requirements.txt
```

**requirements.txt 内容**：
```
torch>=2.0.0
numpy>=1.21.0
matplotlib>=3.4.0
seaborn>=0.11.0
tqdm>=4.62.0
datasets>=2.0.0
transformers>=4.20.0
sacrebleu>=2.0.0
```

### 3. 下载数据集

#### 方法1：自动下载（推荐）

```bash
# 下载PTB数据集（用于语言建模）
python scripts/download_ptb.py --output_dir data/ptb

# 下载Europarl数据集（用于翻译）
python scripts/download_translation_data.py --output_dir data/iwslt2017
```

#### 方法2：手动下载

**PTB数据集**：
1. 访问：https://github.com/wojzaremba/lstm
2. 下载 `ptb.train.txt`, `ptb.valid.txt`, `ptb.test.txt`
3. 放置到 `data/ptb/` 目录

**Europarl数据集**：
1. 访问：https://www.statmt.org/wmt14/translation-task.html#download
2. 下载 `training-parallel-europarl-v7.tgz`
3. 解压并放置到 `data/iwslt2017/training/` 目录

> ? 数据集已备份在项目的压缩包中。

## ? 重现实验（Exact Commands）

### 实验1：PTB语言建模（Encoder-Only）

```bash
python scripts/train_ptb_fixed.py \
    --data_path data/ptb \
    --d_model 256 \
    --n_heads 8 \
    --d_ff 1024 \
    --n_layers 4 \
    --dropout 0.1 \
    --batch_size 32 \
    --epochs 30 \
    --lr 0.0003 \
    --warmup_steps 4000 \
    --max_len 128 \
    --seed 42 \
    --output_dir results/ptb_model_fixed
```

**预期结果**：
- 训练时间：约2-3小时（RTX 3090）
- 最佳验证Loss：约3.67
- 测试Perplexity：约40.8
- 参数量：约15M

**输出文件**：
- `results/ptb_model_fixed/config.json` - 模型配置
- `results/ptb_model_fixed/training_history.json` - 训练历史
- `results/ptb_model_fixed/checkpoints/best_model.pt` - 最佳模型

---

### 实验2：Europarl机器翻译（Encoder-Decoder）

```bash
python scripts/train_translation.py \
    --data_path data/iwslt2017 \
    --src_vocab_size 8000 \
    --tgt_vocab_size 8000 \
    --d_model 256 \
    --n_heads 8 \
    --d_ff 1024 \
    --n_encoder_layers 3 \
    --n_decoder_layers 3 \
    --dropout 0.1 \
    --batch_size 32 \
    --epochs 30 \
    --lr 0.0003 \
    --warmup_steps 4000 \
    --max_len 100 \
    --seed 42 \
    --num_workers 0 \
    --output_dir results/translation
```

**预期结果**：
- 训练时间：约3-4小时（RTX 3090，200K句对）
- 最佳验证Loss：1.93
- 测试Perplexity：6.88
- 测试BLEU：约23.7
- 参数量：约11.7M

**输出文件**：
- `results/translation/config.json`
- `results/translation/training_history.json`
- `results/translation/evaluation_results.json`
- `results/translation/checkpoints/best_model.pt`

---

### 实验3：快速消融实验（5K数据，5 epochs）

**目的**：在小规模数据上快速测试多个超参数的影响

```bash
# 创建5K小数据集
python scripts/create_small_dataset.py \
    --input_dir data/iwslt2017 \
    --output_dir data/iwslt2017_small \
    --num_samples 5000 \
    --seed 42

# 运行快速消融实验（测试4个维度：heads/layers/dropout/size）
python scripts/run_quick_ablation.py \
    --data_path data/iwslt2017_small \
    --base_config_path results/translation/config.json \
    --output_dir results/ablation_quick \
    --epochs 5 \
    --seed 42
```

**预期结果**：
- 总耗时：约4小时（16个实验配置）
- 每个实验：约15分钟
- 关键发现：d_model=512 比 d_model=256 提升6.6%

**测试的超参数组合**：
1. **注意力头数**：1, 2, 4, 8头
2. **模型层数**：1, 2, 3层
3. **Dropout率**：0.0, 0.1, 0.2, 0.3
4. **模型大小**：d_model=256, 512

---

### 实验4：完整数据集模型大小消融（200K数据，15 epochs）

**目的**：在完整数据集上验证模型规模对性能的影响

```bash
python scripts/run_ablation_on_full_data.py \
    --data_path data/iwslt2017 \
    --output_dir results/ablation_full \
    --epochs 15 \
    --seed 42
```

**测试的模型配置**：

| 模型 | d_model | d_ff | n_heads | 参数量 | 命令中的标识 |
|------|---------|------|---------|--------|-------------|
| Small (基准) | 256 | 1024 | 8 | 11.7M | d256_h8_l3_dr0.1 |
| Medium | 512 | 2048 | 8 | 46.5M | d512_h8_l3_dr0.1 |
| Large | 768 | 3072 | 12 | 104M | d768_h12_l3_dr0.1 |

**预期结果**：
- 总耗时：约8.7小时（RTX 3090）
- 每个模型：约2.9小时
- **关键发现**：
  - Medium模型（d=512）：Val Loss **1.888**，比基准提升 **5.7%**
  - Large模型（d=768）：Val Loss **1.873**，比基准提升 **6.4%**
  - 15 epochs的Medium/Large模型已超越30 epochs的基准模型

**为什么选择15 epochs而非30 epochs？**
- 30 epochs预计需要约20小时（3个模型 × 3.5h/模型 × 2倍规模）
- 15 epochs约8-10小时，可在一夜之间完成
- 根据基准模型训练曲线，15 epochs已接近收敛，足以观察性能差异

**输出文件**：
```
results/ablation_full/
├── ablation_results.json          # 总结性结果
├── base_config.json               # 基准配置
└── ablation_size/
    ├── d256_h8_l3_dr0.1/
    │   ├── config.json
    │   ├── training_history.json
    │   └── checkpoints/best_model.pt
    ├── d512_h8_l3_dr0.1/
    │   └── ...
    └── d768_h12_l3_dr0.1/
        └── ...
```

---

### 实验5：生成报告图表

```bash
python scripts/generate_visualizations.py
```

**生成的图表**（保存在 `report/figures/`）：
1. `ptb_training_results.png` - PTB训练曲线
2. `main_training_loss.png` - 翻译任务训练损失
3. `main_training_perplexity.png` - 翻译任务困惑度
4. `ablation_size_full_15epochs.png` - 完整消融对比柱状图
5. `ablation_size_full_curves.png` - 训练曲线对比
6. `ablation_parameter_efficiency.png` - 参数效率分析
7. `attention_visualization.png` - 注意力热力图

---

## ? 实验结果总结

### PTB语言建模（Encoder-Only，4层，30 epochs）

| 指标 | 训练集 | 验证集 | 测试集 |
|------|--------|--------|--------|
| Loss | 3.12 | 3.67 | 3.71 |
| Perplexity | 22.6 | 39.3 | 40.8 |
| Accuracy | 55.8% | 49.3% | 48.7% |

### 机器翻译（Encoder-Decoder，Europarl 200K，30 epochs）

| 指标 | 训练集 | 验证集 | 测试集 |
|------|--------|--------|--------|
| Loss | 1.52 | 1.93 | 1.98 |
| Perplexity | 4.57 | 6.88 | 7.24 |
| BLEU | - | 24.3 | 23.7 |

### 完整消融实验：模型大小影响（200K数据，15 epochs）

| 模型配置 | 参数量 | 最佳验证Loss | vs基准改善 | 训练时间 |
|---------|--------|-------------|-----------|---------|
| Small (d=256) | 11.7M | 2.002 | - (基准) | ~2.9h |
| Medium (d=512) | 46.5M | **1.888** | **+5.7%** | ~2.9h |
| Large (d=768) | 104M | **1.873** | **+6.4%** | ~2.9h |

**关键发现**：
- 模型规模对性能有显著影响，但存在边际递减效应
- Medium模型（d=512）参数效率最优
- 更大的模型收敛更快（15 epochs超越基准30 epochs）

---

## ? 模型架构

### 核心公式

1. **缩放点积注意力**
   ```
   Attention(Q, K, V) = softmax(QK^T / sqrt(d_k)) V
   ```

2. **多头注意力**
   ```
   MultiHead(Q, K, V) = Concat(head_1, ..., head_h) W^O
   where head_i = Attention(QW^Q_i, KW^K_i, VW^V_i)
   ```

3. **位置编码**
   ```
   PE(pos, 2i) = sin(pos / 10000^(2i/d_model))
   PE(pos, 2i+1) = cos(pos / 10000^(2i/d_model))
   ```

4. **前馈网络**
   ```
   FFN(x) = max(0, xW_1 + b_1)W_2 + b_2
   ```

### 默认超参数配置

| 参数 | PTB语言模型 | Europarl翻译 |
|------|-------------|-------------|
| 架构 | Encoder-Only | Encoder-Decoder |
| d_model | 256 | 256 |
| n_heads | 8 | 8 |
| d_ff | 1024 | 1024 |
| Encoder层数 | 4 | 3 |
| Decoder层数 | - | 3 |
| Dropout | 0.1 | 0.1 |
| Batch Size | 32 | 32 |
| Learning Rate | 3e-4 | 3e-4 |
| Warmup Steps | 4000 | 4000 |
| 优化器 | Adam (β1=0.9, β2=0.98) | Adam (β1=0.9, β2=0.98) |
| 梯度裁剪 | 1.0 | 1.0 |
| 随机种子 | 42 | 42 |

---

## ? 代码示例

### 创建Encoder-Only语言模型

```python
import torch
from src.transformer import TransformerForLanguageModeling

# 创建模型
model = TransformerForLanguageModeling(
    vocab_size=10000,
    d_model=256,
    n_heads=8,
    d_ff=1024,
    n_layers=4,
    max_len=512,
    dropout=0.1
)

# 前向传播
x = torch.randint(0, 10000, (32, 100))  # [batch_size, seq_len]
output = model(x)  # [batch_size, seq_len, vocab_size]

print(f"参数量: {sum(p.numel() for p in model.parameters()):,}")
```

### 创建Encoder-Decoder翻译模型

```python
from src.transformer import Transformer

# 创建模型
model = Transformer(
    src_vocab_size=8000,
    tgt_vocab_size=8000,
    d_model=256,
    n_heads=8,
    d_ff=1024,
    n_encoder_layers=3,
    n_decoder_layers=3,
    max_len=100,
    dropout=0.1
)

# 前向传播
src = torch.randint(0, 8000, (32, 50))  # 源语言
tgt = torch.randint(0, 8000, (32, 50))  # 目标语言
output = model(src, tgt)  # [batch_size, tgt_len, tgt_vocab_size]
```

### 加载预训练模型

```python
import torch

# 加载检查点
checkpoint = torch.load('results/translation/checkpoints/best_model.pt')

# 创建模型并加载权重
model = Transformer(...)
model.load_state_dict(checkpoint['model_state_dict'])

# 评估模式
model.eval()
with torch.no_grad():
    output = model(src, tgt)
```

---

## ? 消融实验详细说明

### 两阶段实验策略

#### 阶段一：快速消融（5K数据，5 epochs）
- **目的**：快速测试多个超参数的影响
- **数据规模**：5K句对（2.5%的完整数据）
- **训练时长**：每个实验约15分钟
- **测试维度**：
  1. 注意力头数（1, 2, 4, 8）
  2. 模型层数（1, 2, 3）
  3. Dropout率（0.0, 0.1, 0.2, 0.3）
  4. 模型大小（d_model=256, 512）
- **总耗时**：约4小时

#### 阶段二：完整消融（200K数据，15 epochs）
- **目的**：在完整数据集上验证关键因素
- **选择原因**：快速实验显示模型大小影响最显著（6.6%提升）
- **测试配置**：d_model = 256, 512, 768
- **训练时长**：每个模型约2.9小时
- **总耗时**：8.69小时

**为什么用15 epochs而非30 epochs？**
- 资源优化：15 epochs需8.7h，30 epochs需20h
- 效果充分：根据基准模型曲线，15 epochs已接近收敛
- 趋势明确：足以观察不同模型规模的性能差异

---

## ? 参考文献

1. Vaswani, A., et al. (2017). *Attention is all you need*. NeurIPS.
2. Devlin, J., et al. (2018). *BERT: Pre-training of deep bidirectional transformers*. arXiv:1810.04805.
3. Radford, A., et al. (2019). *Language models are unsupervised multitask learners*. OpenAI Blog.
4. Marcus, M. P., et al. (1993). *Building a large annotated corpus of English: The Penn Treebank*. Computational Linguistics.
5. Koehn, P. (2005). *Europarl: A parallel corpus for statistical machine translation*. MT Summit.

---

## ? 贡献与反馈

本项目为课程作业，主要用于学习和研究目的。如有问题或建议，欢迎：
- 提交Issue
- 发送邮件至：[your-email]
- 参考LaTeX报告获取更多技术细节

---

## ? 许可证

本项目仅用于教育和学术目的。代码遵循MIT许可证。

---

## ??? 作者信息

**姓名**：兰逸凡  
**学号**：25115072  
**课程**：M502082B《大模型基础与应用》  
**时间**：2025年11月

---

## ? 致谢

- PyTorch团队提供的优秀深度学习框架
- Hugging Face提供的数据集工具
- Vaswani等人的开创性Transformer工作
- 课程老师的悉心指导

---

**注意**：本项目是从零实现Transformer的教学项目，主要目的是深入理解其内部机制。生产环境请使用Hugging Face Transformers等成熟库。
