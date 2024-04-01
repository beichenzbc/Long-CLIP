#!/usr/bin/env bash
export MKL_NUM_THREADS=1
export NUMEXPR_NUM_THREADS=1
export OMP_NUM_THREADS=1
set -x

PARTITION=your_partition
JOB_NAME=long-clip
GPUS=16
GPUS_PER_NODE=8
CPUS_PER_TASK=16
PYTHONPATH="$(dirname $0)/..":$PYTHONPATH \
srun -p ${PARTITION} \
    --job-name=${JOB_NAME} \
    --gres=gpu:${GPUS_PER_NODE} \
    --ntasks=${GPUS} \
    --ntasks-per-node=${GPUS_PER_NODE} \
    --cpus-per-task=${CPUS_PER_TASK} \
    --quotatype=reserved \
    --kill-on-bad-exit=1 \
    --async \
    --output='longclip.out' \
    --error='longclip.err' \
    ${SRUN_ARGS} \
    python -u train.py