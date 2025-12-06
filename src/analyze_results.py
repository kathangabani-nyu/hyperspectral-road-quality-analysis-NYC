"""
Analyze classification results and identify areas for improvement.
"""
import numpy as np
import rasterio
import os
import advanced_config as config

def analyze_classification_map(map_path):
    """Analyze classification map statistics."""
    if not os.path.exists(map_path):
        print(f"File not found: {map_path}")
        return None
    
    with rasterio.open(map_path) as src:
        data = src.read(1)
        height, width = data.shape
        
        # Statistics
        total_pixels = height * width
        pavement_pixels = (data > 0).sum()
        pavement_percent = (pavement_pixels / total_pixels) * 100
        
        # Connected components
        from scipy.ndimage import label
        labeled, n_regions = label(data > 0)
        
        # Region sizes
        region_sizes = []
        for i in range(1, n_regions + 1):
            region_size = (labeled == i).sum()
            region_sizes.append(region_size)
        
        region_sizes = np.array(region_sizes)
        
        print(f"\n{'='*70}")
        print(f"CLASSIFICATION MAP ANALYSIS: {os.path.basename(map_path)}")
        print(f"{'='*70}")
        print(f"Image size: {height} x {width} = {total_pixels:,} pixels")
        print(f"Pavement pixels: {pavement_pixels:,} ({pavement_percent:.2f}%)")
        print(f"Number of regions: {n_regions}")
        
        if len(region_sizes) > 0:
            print(f"\nRegion size statistics:")
            print(f"  Smallest: {region_sizes.min():,} pixels")
            print(f"  Largest: {region_sizes.max():,} pixels")
            print(f"  Mean: {region_sizes.mean():.0f} pixels")
            print(f"  Median: {np.median(region_sizes):.0f} pixels")
            
            # Count regions by size
            small_regions = (region_sizes < 100).sum()
            medium_regions = ((region_sizes >= 100) & (region_sizes < 1000)).sum()
            large_regions = (region_sizes >= 1000).sum()
            
            print(f"\nRegion size distribution:")
            print(f"  Small (<100 pixels): {small_regions}")
            print(f"  Medium (100-1000 pixels): {medium_regions}")
            print(f"  Large (>=1000 pixels): {large_regions}")
        
        return {
            'total_pixels': total_pixels,
            'pavement_pixels': pavement_pixels,
            'pavement_percent': pavement_percent,
            'n_regions': n_regions,
            'region_sizes': region_sizes
        }

def compare_maps(map1_path, map2_path):
    """Compare two classification maps."""
    stats1 = analyze_classification_map(map1_path)
    stats2 = analyze_classification_map(map2_path)
    
    if stats1 and stats2:
        print(f"\n{'='*70}")
        print("COMPARISON")
        print(f"{'='*70}")
        print(f"Pavement pixels: {stats1['pavement_pixels']:,} -> {stats2['pavement_pixels']:,}")
        print(f"Change: {stats2['pavement_pixels'] - stats1['pavement_pixels']:,} pixels "
              f"({(stats2['pavement_pixels'] - stats1['pavement_pixels']) / stats1['pavement_pixels'] * 100:.1f}%)")
        print(f"Regions: {stats1['n_regions']} -> {stats2['n_regions']}")

if __name__ == "__main__":
    # Analyze latest classification
    latest_map = os.path.join(config.RESULTS_DIR, "pavement_classification_advanced.tif")
    analyze_classification_map(latest_map)

