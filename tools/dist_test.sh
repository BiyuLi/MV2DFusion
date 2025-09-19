#!/usr/bin/env bash

# 读取核心参数
CONFIG=$1
CHECKPOINT=$2
GPUS=$3

# 分布式默认参数
NNODES=${NNODES:-1}
NODE_RANK=${NODE_RANK:-0}
PORT=${PORT:-29500}
MASTER_ADDR=${MASTER_ADDR:-"127.0.0.1"}

# 配置Python路径
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
# 使用torchrun（替代过时的torch.distributed.launch）
torchrun \
    --nnodes=$NNODES \
    --node_rank=$NODE_RANK \
    --master_addr=$MASTER_ADDR \
    --nproc_per_node=$GPUS \
    --master_port=$PORT \
    "$(dirname "$0")/test.py" \  # 调用test.py（确保路径正确）
    "$CONFIG" \  # 传递核心参数：配置文件路径
    "$CHECKPOINT" \  # 传递核心参数：权重文件路径
    --launcher pytorch \
    "${@:4}"  # 传递剩余参数（如--format-only等）