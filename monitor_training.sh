#!/bin/bash
# Monitor training progress and report when target is reached

STAGE_DIR="/home/workspace/projects/transformer/UPT/benchmarking/save/stage1"
TARGET_REL_L1=0.001  # 0.1%

echo "Monitoring training progress..."
echo "Target: rel_l1 < 0.1% (0.001)"
echo ""

while true; do
    # Find most recent stage
    LATEST_STAGE=$(ls -t $STAGE_DIR | head -1)
    ENTRIES_FILE="$STAGE_DIR/$LATEST_STAGE/primitive/entries.yaml"
    
    if [ -f "$ENTRIES_FILE" ]; then
        # Extract latest metrics
        LATEST_EPOCH=$(grep -A 1000 "loss/online/x_hat/E100:" "$ENTRIES_FILE" | grep "^  [0-9]" | tail -1 | awk '{print $1}' | tr -d ':')
        LATEST_LOSS=$(grep -A 1000 "loss/online/x_hat/E100:" "$ENTRIES_FILE" | grep "^  [0-9]" | tail -1 | awk '{print $2}')
        LATEST_REL_L1=$(grep -A 1000 "loss/online/rel_l1/E100:" "$ENTRIES_FILE" | grep "^  [0-9]" | tail -1 | awk '{print $2}')
        
        if [ ! -z "$LATEST_EPOCH" ]; then
            echo "[$(date +%H:%M:%S)] Epoch $LATEST_EPOCH: Loss=$LATEST_LOSS, Rel L1=$LATEST_REL_L1"
            
            # Check if target reached
            if [ ! -z "$LATEST_REL_L1" ]; then
                TARGET_REACHED=$(python3 -c "print(1 if float('$LATEST_REL_L1') < $TARGET_REL_L1 else 0)")
                if [ "$TARGET_REACHED" = "1" ]; then
                    echo ""
                    echo "✓✓✓ TARGET REACHED! ✓✓✓"
                    echo "Stage: $LATEST_STAGE"
                    echo "Epoch: $LATEST_EPOCH"
                    echo "Rel L1: $LATEST_REL_L1 (< 0.1%)"
                    echo ""
                    exit 0
                fi
            fi
        fi
    fi
    
    sleep 120  # Check every 2 minutes
done



