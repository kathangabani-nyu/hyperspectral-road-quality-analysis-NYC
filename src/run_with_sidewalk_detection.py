"""
Quick script to run the enhanced pipeline with sidewalk detection enabled.
This will be executed after manual mask is created.
"""
import subprocess
import sys
import os

print("="*70)
print("RUNNING ENHANCED PIPELINE WITH SIDEWALK DETECTION")
print("="*70)
print("\nConfiguration:")
print("  - Sidewalk-specific filtering: ENABLED")
print("  - Manual mask integration: ENABLED")
print()

# Check if manual mask exists
mask_path = "outputs_research/manual_pavement_mask.npy"
if os.path.exists(mask_path):
    print(f"Found manual mask: {mask_path}")
    import numpy as np
    mask = np.load(mask_path)
    print(f"  Pavement pixels: {mask.sum()} ({mask.sum() / mask.size * 100:.2f}%)")
else:
    print(f"WARNING: Manual mask not found: {mask_path}")
    print("  Pipeline will run with sidewalk filtering only")
    print("  (Manual mask will be skipped)")

print("\n" + "="*70)
print("Starting enhanced pipeline...")
print("="*70 + "\n")

# Run main_enhanced.py
subprocess.run([sys.executable, "main_enhanced.py"])

