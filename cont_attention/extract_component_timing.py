#!/usr/bin/env python
"""
Extract and display component timing for UPT simformer model from training logs.
"""
import sys
import re

def parse_log_for_timing(log_content):
    """Parse log content to extract component timings."""
    
    # Find the profiling section
    lines = log_content.split('\n')
    in_profiling = False
    timings = {}
    
    for line in lines:
        if 'full profiling times:' in line:
            in_profiling = True
            continue
        
        if in_profiling:
            # Look for encoder, latent, decoder lines
            if 'model.encoder' in line and 'geometry_encoder' not in line:
                match = re.search(r'(\d+\.\d+)\s+.*model\.encoder', line)
                if match:
                    timings['encoder'] = float(match.group(1))
            elif 'model.latent' in line:
                match = re.search(r'(\d+\.\d+)\s+.*model\.latent', line)
                if match:
                    timings['latent'] = float(match.group(1))
            elif 'model.decoder' in line:
                match = re.search(r'(\d+\.\d+)\s+.*model\.decoder', line)
                if match:
                    timings['decoder'] = float(match.group(1))
    
    return timings

def display_timing_summary(timings):
    """Display timing summary in a nice format."""
    if not timings:
        print("No timing information found in log.")
        return
    
    total = sum(timings.values())
    
    print("\n" + "="*60)
    print("UPT Simformer Component Timing Summary")
    print("="*60)
    print(f"  Encoder:      {timings.get('encoder', 0.0)*1000:.2f} ms ({timings.get('encoder', 0.0)/total*100:.1f}%)" if total > 0 else f"  Encoder:      {timings.get('encoder', 0.0)*1000:.2f} ms")
    print(f"  Approximator: {timings.get('latent', 0.0)*1000:.2f} ms ({timings.get('latent', 0.0)/total*100:.1f}%)" if total > 0 else f"  Approximator: {timings.get('latent', 0.0)*1000:.2f} ms")
    print(f"  Decoder:      {timings.get('decoder', 0.0)*1000:.2f} ms ({timings.get('decoder', 0.0)/total*100:.1f}%)" if total > 0 else f"  Decoder:      {timings.get('decoder', 0.0)*1000:.2f} ms")
    print(f"  Total:        {total*1000:.2f} ms")
    print("="*60 + "\n")

if __name__ == "__main__":
    if len(sys.argv) > 1:
        # Read from file
        with open(sys.argv[1], 'r') as f:
            log_content = f.read()
    else:
        # Read from stdin
        log_content = sys.stdin.read()
    
    timings = parse_log_for_timing(log_content)
    display_timing_summary(timings)



