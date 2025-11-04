"""Monitor quadtree-UPT training progress"""

import sys
import time

def monitor():
    log_file = "train_quadtree.log"
    
    print("Monitoring training (Ctrl+C to stop)")
    print("=" * 80)
    
    try:
        with open(log_file, 'r') as f:
            # Skip to end
            f.seek(0, 2)
            
            while True:
                line = f.readline()
                if line:
                    # Filter out excessive warnings
                    if "WARNING: Sample" not in line or line.count("WARNING") < 2:
                        print(line, end='')
                else:
                    time.sleep(0.5)
    except KeyboardInterrupt:
        print("\nStopped monitoring")
    except FileNotFoundError:
        print(f"Log file not found: {log_file}")

if __name__ == "__main__":
    monitor()

