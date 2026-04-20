#!/usr/bin/env bash

INTERVAL=600  # 10 minutes in seconds
OUTPUT="gpu_usage_$(date +%Y%m%d_%H%M%S).csv"

echo "timestamp,gpu_index,gpu_name,vram_used_mb,vram_total_mb,vram_used_pct,gpu_util_pct" > "$OUTPUT"

echo "Logging GPU usage to: $OUTPUT"
echo "Press Ctrl+C to stop."

while true; do
    TIMESTAMP=$(date +"%Y-%m-%d %H:%M:%S")

    # Query: index, name, memory.used, memory.total, utilization.gpu
    nvidia-smi --query-gpu=index,name,memory.used,memory.total,utilization.gpu \
               --format=csv,noheader,nounits | \
    while IFS=',' read -r idx name mem_used mem_total gpu_util; do
        # Trim whitespace
        idx=$(echo "$idx" | tr -d ' ')
        name=$(echo "$name" | tr -d ' ')
        mem_used=$(echo "$mem_used" | tr -d ' ')
        mem_total=$(echo "$mem_total" | tr -d ' ')
        gpu_util=$(echo "$gpu_util" | tr -d ' ')

        if [[ "$mem_total" -gt 0 ]]; then
            vram_pct=$(awk "BEGIN {printf \"%.1f\", $mem_used / $mem_total * 100}")
        else
            vram_pct="N/A"
        fi

        echo "\"$TIMESTAMP\",$idx,\"$name\",$mem_used,$mem_total,$vram_pct,$gpu_util"
    done >> "$OUTPUT"

    sleep "$INTERVAL"
done
