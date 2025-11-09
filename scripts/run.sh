#!/bin/bash
# Transformer模型训练脚本
# 用法: bash scripts/run.sh [task]
# 任务选项: ptb, translation, quick_ablation, full_ablation, visualize

set -e  # 遇到错误立即退出

# 颜色输出
RED='\033[0;31m'
GREEN='\033[0;32m'
YELLOW='\033[1;33m'
NC='\033[0m' # No Color

# 默认参数
SEED=42
DATA_DIR="data"
RESULTS_DIR="results"

# 打印带颜色的消息
print_info() {
    echo -e "${GREEN}[INFO]${NC} $1"
}

print_warning() {
    echo -e "${YELLOW}[WARNING]${NC} $1"
}

print_error() {
    echo -e "${RED}[ERROR]${NC} $1"
}

# 检查GPU是否可用
check_gpu() {
    if command -v nvidia-smi &> /dev/null; then
        print_info "GPU检测成功:"
        nvidia-smi --query-gpu=name,memory.total --format=csv,noheader
    else
        print_warning "未检测到GPU，将使用CPU训练（速度较慢）"
    fi
}

# PTB语言建模训练
train_ptb() {
    print_info "开始训练PTB语言模型（Encoder-Only）..."
    python scripts/train_ptb_fixed.py \
        --data_path ${DATA_DIR}/ptb \
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
        --seed ${SEED} \
        --output_dir ${RESULTS_DIR}/ptb_model_fixed
    
    print_info "PTB模型训练完成！结果保存在 ${RESULTS_DIR}/ptb_model_fixed/"
}

# 机器翻译训练
train_translation() {
    print_info "开始训练Europarl机器翻译模型（Encoder-Decoder）..."
    python scripts/train_translation.py \
        --data_path ${DATA_DIR}/iwslt2017 \
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
        --seed ${SEED} \
        --num_workers 0 \
        --output_dir ${RESULTS_DIR}/translation
    
    print_info "翻译模型训练完成！结果保存在 ${RESULTS_DIR}/translation/"
}

# 快速消融实验
run_quick_ablation() {
    print_info "开始快速消融实验（5K数据，5 epochs）..."
    
    # 创建小数据集（如果不存在）
    if [ ! -d "${DATA_DIR}/iwslt2017_small" ]; then
        print_info "创建5K小数据集..."
        python scripts/create_small_dataset.py \
            --input_dir ${DATA_DIR}/iwslt2017 \
            --output_dir ${DATA_DIR}/iwslt2017_small \
            --num_samples 5000 \
            --seed ${SEED}
    fi
    
    # 运行快速消融
    python scripts/run_quick_ablation.py \
        --data_path ${DATA_DIR}/iwslt2017_small \
        --base_config_path ${RESULTS_DIR}/translation/config.json \
        --output_dir ${RESULTS_DIR}/ablation_quick \
        --epochs 5 \
        --seed ${SEED}
    
    print_info "快速消融实验完成！结果保存在 ${RESULTS_DIR}/ablation_quick/"
}

# 完整消融实验
run_full_ablation() {
    print_info "开始完整消融实验（200K数据，15 epochs）..."
    print_warning "预计运行时间：约8-10小时"
    
    python scripts/run_ablation_on_full_data.py \
        --data_path ${DATA_DIR}/iwslt2017 \
        --output_dir ${RESULTS_DIR}/ablation_full \
        --epochs 15 \
        --seed ${SEED}
    
    print_info "完整消融实验完成！结果保存在 ${RESULTS_DIR}/ablation_full/"
}

# 生成可视化
generate_visualizations() {
    print_info "生成实验结果可视化..."
    python scripts/generate_visualizations.py
    print_info "可视化图表已生成！保存在 report/figures/"
}

# 主函数
main() {
    echo "=================================================="
    echo "  Transformer模型训练脚本"
    echo "  课程: M502082B《大模型基础与应用》"
    echo "=================================================="
    echo ""
    
    # 检查GPU
    check_gpu
    echo ""
    
    # 解析命令行参数
    TASK=${1:-"help"}
    
    case $TASK in
        ptb)
            train_ptb
            ;;
        translation)
            train_translation
            ;;
        quick_ablation)
            run_quick_ablation
            ;;
        full_ablation)
            run_full_ablation
            ;;
        visualize)
            generate_visualizations
            ;;
        all)
            print_info "运行完整实验流程..."
            train_ptb
            train_translation
            run_quick_ablation
            run_full_ablation
            generate_visualizations
            print_info "所有实验完成！"
            ;;
        help|*)
            echo "用法: bash scripts/run.sh [task]"
            echo ""
            echo "可用任务:"
            echo "  ptb              - 训练PTB语言模型（Encoder-Only，约2-3小时）"
            echo "  translation      - 训练Europarl翻译模型（Encoder-Decoder，约3-4小时）"
            echo "  quick_ablation   - 快速消融实验（5K数据，约4小时）"
            echo "  full_ablation    - 完整消融实验（200K数据，约8-10小时）"
            echo "  visualize        - 生成可视化图表"
            echo "  all              - 运行所有实验（总计约20小时）"
            echo ""
            echo "示例:"
            echo "  bash scripts/run.sh ptb              # 训练PTB模型"
            echo "  bash scripts/run.sh translation      # 训练翻译模型"
            echo "  bash scripts/run.sh quick_ablation   # 快速消融"
            echo ""
            echo "环境变量:"
            echo "  SEED=42          - 随机种子（默认: 42）"
            echo "  DATA_DIR=data    - 数据目录（默认: data）"
            echo "  RESULTS_DIR=results - 结果目录（默认: results）"
            echo ""
            echo "完整重现实验:"
            echo "  bash scripts/run.sh all"
            ;;
    esac
}

# 运行主函数
main "$@"
