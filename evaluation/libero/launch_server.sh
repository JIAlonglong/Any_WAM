
save_root='visualization/'
mkdir -p $save_root

# 使用方法:
#   bash evaluation/libero/launch_server.sh                    # 使用默认 checkpoint
#   bash evaluation/libero/launch_server.sh /path/to/checkpoint  # 指定 checkpoint
#   NUM_STEPS=4 bash evaluation/libero/launch_server.sh       # 指定推理步数

CHECKPOINT_PATH="${1:-distillation_flowmap/output_libero/checkpoints/step_1000/online_student/transformer}"
NUM_STEPS="${NUM_STEPS:-2}"

python -m torch.distributed.run \
    --nproc_per_node 1 \
    --master_port 29061 \
    wan_va/wan_va_server.py \
    --config-name libero \
    --port 29056 \
    --checkpoint-path "$CHECKPOINT_PATH" \
    --num-steps "$NUM_STEPS" \
    --save_root $save_root
