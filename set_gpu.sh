#!/bin/bash
# GPU Selector for TARS
# Automatically finds GPU with lowest usage and sets CUDA_VISIBLE_DEVICES

echo "=========================================="
echo "🎯 TARS GPU Selector"
echo "=========================================="

# Check if we're on a GPU node
if ! command -v nvidia-smi &> /dev/null; then
    echo "❌ nvidia-smi not found. Are you on a GPU node?"
    echo "   Request one with: srun --gpus=1 --time=02:00:00 --pty bash"
    exit 1
fi

# Show available GPUs
echo -e "\n📊 Available GPUs:"
nvidia-smi --query-gpu=index,name,memory.used,memory.total,memory.free --format=csv | tail -n +2 | while IFS=, read -r idx name used total free; do
    # Calculate usage percentage
    used_num=$(echo $used | awk '{print $1}')
    total_num=$(echo $total | awk '{print $1}')
    if [ "$total_num" != "0" ] && [ "$total_num" != "" ]; then
        pct=$((used_num * 100 / total_num))
        echo "   GPU $idx: $name - ${used_num}MB / ${total_num}MB (${pct}% used) - ${free}MB free"
    fi
done

# Get GPU indices and memory usage, find the one with most free memory
BEST_GPU=$(nvidia-smi --query-gpu=index,memory.free --format=csv,noheader,nounits 2>/dev/null | sort -t, -k2 -rn | head -1 | cut -d, -f1)

if [ -z "$BEST_GPU" ]; then
    echo "❌ Could not determine best GPU, defaulting to GPU 0"
    BEST_GPU=0
fi

# Get usage info for selected GPU
USAGE=$(nvidia-smi --query-gpu=memory.used,memory.total --format=csv,noheader,nounits --id=$BEST_GPU 2>/dev/null)
USED=$(echo $USAGE | cut -d, -f1 | xargs)
TOTAL=$(echo $USAGE | cut -d, -f2 | xargs)
NAME=$(nvidia-smi --query-gpu=name --format=csv,noheader --id=$BEST_GPU 2>/dev/null)

if [ -n "$USED" ] && [ -n "$TOTAL" ]; then
    PCT=$((USED * 100 / TOTAL))
    FREE=$((TOTAL - USED))
    echo -e "\n🎯 Selected GPU: $BEST_GPU ($NAME)"
    echo "   Memory: ${USED}MB / ${TOTAL}MB (${PCT}% used)"
    echo "   Free:   ${FREE}MB"
fi

# Set the environment variable
export CUDA_VISIBLE_DEVICES=$BEST_GPU