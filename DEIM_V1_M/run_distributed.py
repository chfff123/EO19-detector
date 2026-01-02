#!/bin/bash

# 强制设置关键环境变量
export NCCL_DEBUG=INFO
export NCCL_ASYNC_ERROR_HANDLING=1
export NCCL_SOCKET_IFNAME=eth0  # 替换为实际网卡名
export OMP_NUM_THREADS=1
export CUDA_LAUNCH_BLOCKING=1
export PYTHONFAULTHANDLER=1
export TORCH_DISTRIBUTED_DEBUG=DETAIL  # PyTorch分布式调试

# 设置NCCL参数避免常见问题
export NCCL_IB_DISABLE=1           # 禁用InfiniBand
export NCCL_P2P_DISABLE=1          # 禁用P2P通信
export NCCL_SHM_DISABLE=1          # 禁用共享内存
export NCCL_IGNORE_CPU_AFFINITY=1  # 忽略CPU亲和性

# 使用固定端口避免随机端口冲突
MASTER_PORT=29500

# 获取主机IP
MASTER_ADDR=$(hostname -I | awk '{print $1}')

# 使用torchrun启动
torchrun \
    --nnodes=1 \
    --nproc_per_node=4 \             # 根据实际GPU数量调整
    --rdzv_id=deim_exp_$(date +%s) \ # 唯一ID带时间戳
    --rdzv_backend=c10d \
    --rdzv_endpoint=$MASTER_ADDR:$MASTER_PORT \
    --max_restarts=0 \               # 禁用自动重启以获取真实错误
    train.py \
    --distributed \
    --env_name=HalfCheetah-v3 \
    --mem_size=1000000 \
    --agent=sac \
    --seed=0