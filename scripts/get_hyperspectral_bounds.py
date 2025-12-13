#!/usr/bin/env python3
"""
Extract spatial bounds from hyperspectral data file.

This script calculates the geographic bounds (minx, miny, maxx, maxy) of the
hyperspectral dataset, optionally accounting for a specific window.

Usage:
    python scripts/get_hyperspectral_bounds.py [--config advanced_config] [--window x y width height]
    
Output:
    Prints bounds in format: minx,miny,maxx,maxy
    Also saves to bounds.txt for use in other scripts.
"""

import sys
import os
import argparse
from pathlib import Path

# Add src to path
project_root = Path(__file__).parent.parent
sys.path.insert(0, str(project_root))
sys.path.insert(0, str(project_root / "src"))

try:
    import rasterio
    from rasterio.transform import xy
except ImportError:
    print("ERROR: rasterio not installed. Install with: pip install rasterio")
    sys.exit(1)


def get_bounds(hyperspectral_path, window=None):
    """
    Get geographic bounds from hyperspectral file.
    
    Parameters:
    -----------
    hyperspectral_path : str
        Path to hyperspectral .pix file
    window : tuple, optional
        (col_off, row_off, width, height) window to extract bounds for
        
    Returns:
    --------
    bounds : tuple
        (minx, miny, maxx, maxy) in geographic coordinates
    crs : CRS
        Coordinate reference system
    transform : Affine
        Transform matrix
    """
    if not os.path.exists(hyperspectral_path):
        print(f"ERROR: File not found: {hyperspectral_path}")
        sys.exit(1)
    
    with rasterio.open(hyperspectral_path) as src:
        crs = src.crs
        width = src.width
        height = src.height
        
        if window is not None:
            from rasterio.windows import Window
            col_off, row_off, win_width, win_height = window
            window_obj = Window(col_off, row_off, win_width, win_height)
            transform = src.window_transform(window_obj)
            width = win_width
            height = win_height
        else:
            transform = src.transform
        
        # Calculate bounds
        # Bottom-left corner
        minx, maxy = xy(transform, 0, height)
        # Top-right corner
        maxx, miny = xy(transform, width, 0)
        
        bounds = (minx, miny, maxx, maxy)
        
        return bounds, crs, transform


def main():
    parser = argparse.ArgumentParser(
        description="Extract spatial bounds from hyperspectral data"
    )
    parser.add_argument(
        "--config",
        choices=["config", "advanced_config"],
        default="advanced_config",
        help="Configuration module to use (default: advanced_config)"
    )
    parser.add_argument(
        "--file",
        type=str,
        help="Path to hyperspectral file (overrides config)"
    )
    parser.add_argument(
        "--window",
        type=int,
        nargs=4,
        metavar=("X", "Y", "WIDTH", "HEIGHT"),
        help="Window coordinates: start_x start_y width height"
    )
    parser.add_argument(
        "--output",
        type=str,
        help="Output file for bounds (default: bounds.txt)"
    )
    
    args = parser.parse_args()
    
    # Get hyperspectral path
    if args.file:
        hyperspectral_path = args.file
    else:
        # Import config
        try:
            if args.config == "advanced_config":
                import advanced_config as config
            else:
                import config as config
            hyperspectral_path = config.HYPERSPECTRAL_PATH
            
            # Check if window should be used from config
            if args.window is None:
                start_x = getattr(config, 'START_X', None)
                start_y = getattr(config, 'START_Y', None)
                test_size_x = getattr(config, 'TEST_SIZE_X', None)
                test_size_y = getattr(config, 'TEST_SIZE_Y', None)
                
                if all(v is not None for v in [start_x, start_y, test_size_x, test_size_y]):
                    args.window = (start_x, start_y, test_size_x, test_size_y)
                    print(f"Using window from config: {args.window}")
        except ImportError as e:
            print(f"ERROR: Could not import config: {e}")
            sys.exit(1)
    
    # Resolve path relative to project root
    if not os.path.isabs(hyperspectral_path):
        hyperspectral_path = os.path.join(project_root, hyperspectral_path)
    
    # Get bounds
    try:
        bounds, crs, transform = get_bounds(hyperspectral_path, args.window)
        minx, miny, maxx, maxy = bounds
    except Exception as e:
        print(f"ERROR: Failed to read bounds: {e}")
        sys.exit(1)
    
    # Print results
    print("\n" + "="*60)
    print("HYPERSPECTRAL BOUNDS")
    print("="*60)
    print(f"File: {hyperspectral_path}")
    if args.window:
        print(f"Window: {args.window}")
    print(f"CRS: {crs}")
    print(f"\nBounds (minx, miny, maxx, maxy):")
    print(f"  {minx:.6f}, {miny:.6f}, {maxx:.6f}, {maxy:.6f}")
    print(f"\nBounds (comma-separated):")
    print(f"{minx:.6f},{miny:.6f},{maxx:.6f},{maxy:.6f}")
    
    # Save to file
    output_file = args.output or "bounds.txt"
    with open(output_file, 'w') as f:
        f.write(f"# Hyperspectral bounds\n")
        f.write(f"# File: {hyperspectral_path}\n")
        if args.window:
            f.write(f"# Window: {args.window}\n")
        f.write(f"# CRS: {crs}\n")
        f.write(f"# Format: minx,miny,maxx,maxy\n")
        f.write(f"{minx:.6f},{miny:.6f},{maxx:.6f},{maxy:.6f}\n")
    
    print(f"\nBounds saved to: {output_file}")
    print("="*60)
    
    return 0


if __name__ == "__main__":
    sys.exit(main())

