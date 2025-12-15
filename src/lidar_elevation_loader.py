"""
LiDAR elevation data loading and height-based filtering.

This module handles:
- Loading LiDAR point cloud data (.laz files)
- Converting to raster DEM (Digital Elevation Model)
- Aligning with hyperspectral data
- Calculating relative height offsets
- Filtering based on height differences (10cm threshold)
"""

import numpy as np
import os
import glob
import rasterio
from rasterio.warp import reproject, Resampling
from rasterio.windows import Window, from_bounds
from rasterio.transform import xy, from_bounds as transform_from_bounds
from rasterio.crs import CRS
import warnings
warnings.filterwarnings('ignore')

try:
    import laspy
except ImportError:
    laspy = None
    print("WARNING: laspy not installed. Install with: pip install laspy")

try:
    import advanced_config as config
except ImportError:
    import config


def find_lidar_files(lidar_dir):
    """
    Find all .laz files in the LiDAR directory.
    
    Parameters:
    -----------
    lidar_dir : str
        Path to directory containing .laz files
        
    Returns:
    --------
    laz_files : list
        List of paths to .laz files
    """
    if not os.path.exists(lidar_dir):
        raise FileNotFoundError(f"LiDAR directory not found: {lidar_dir}")
    
    laz_files = glob.glob(os.path.join(lidar_dir, "*.laz"))
    if not laz_files:
        # Try .las files
        laz_files = glob.glob(os.path.join(lidar_dir, "*.las"))
    
    if not laz_files:
        raise FileNotFoundError(f"No .laz or .las files found in {lidar_dir}")
    
    if config.VERBOSE:
        print(f"Found {len(laz_files)} LiDAR file(s)")
    
    return sorted(laz_files)


def load_lidar_points(laz_files, bounds=None, bounds_crs=None, diagnostic_mode=False):
    """
    Load LiDAR points from .laz files, optionally filtered by bounds.
    
    Parameters:
    -----------
    laz_files : list
        List of paths to .laz files
    bounds : tuple, optional
        (minx, miny, maxx, maxy) bounds to filter points (in bounds_crs)
    bounds_crs : CRS, optional
        CRS of bounds (will transform to point CRS if different)
    diagnostic_mode : bool
        If True, load points without bounds first to diagnose coverage
        
    Returns:
    --------
    points : ndarray
        Array of shape (n_points, 3) with columns [x, y, z]
    point_crs : CRS
        CRS of the points
    """
    if laspy is None:
        raise ImportError("laspy is required to read .laz files. Install with: pip install laspy")
    
    all_points = []
    point_crs = None
    file_bounds_info = []  # Store bounds info for diagnostics
    
    # Try to detect CRS from first file's coordinate ranges
    # NYC LiDAR is typically in EPSG:2263 (NAD83 / New York Long Island, feet)
    # Typical ranges: X ~900k-1M, Y ~100k-300k
    default_lidar_crs = CRS.from_epsg(2263)  # NAD83 / New York Long Island (ftUS)
    
    # First pass: Load without bounds to detect CRS and get file coverage
    if diagnostic_mode or (bounds is not None):
        for laz_file in laz_files:
            try:
                try:
                    import lazrs
                    if hasattr(laspy, 'LazBackend'):
                        las = laspy.read(laz_file, laz_backend=laspy.LazBackend.Lazrs)
                    else:
                        las = laspy.read(laz_file, laz_backend='lazrs')
                except (ImportError, AttributeError, TypeError, ValueError):
                    las = laspy.read(laz_file)
                
                if point_crs is None:
                    x_sample = las.x[:1000] if len(las.x) > 1000 else las.x
                    y_sample = las.y[:1000] if len(las.y) > 1000 else las.y
                    
                    x_mean = np.mean(x_sample)
                    y_mean = np.mean(y_sample)
                    x_min, x_max = np.min(las.x), np.max(las.x)
                    y_min, y_max = np.min(las.y), np.max(las.y)
                    
                    # EPSG:2263 (NY State Plane, feet): X ~900k-1M, Y ~100k-300k
                    # EPSG:3178 (Albers, meters): X ~500k-600k, Y ~4.4M-4.5M
                    if 900000 < x_mean < 1100000 and 100000 < y_mean < 400000:
                        point_crs = CRS.from_epsg(2263)
                        if config.VERBOSE:
                            print(f"    Detected CRS: EPSG:2263 (State Plane, feet) from coordinates")
                    elif 400000 < x_mean < 700000 and 4000000 < y_mean < 5000000:
                        point_crs = CRS.from_epsg(3178)
                        if config.VERBOSE:
                            print(f"    Detected CRS: EPSG:3178 (Albers, meters) from coordinates")
                    else:
                        point_crs = default_lidar_crs
                        if config.VERBOSE:
                            print(f"    Using default CRS: EPSG:2263 (coordinates: X={x_mean:.0f}, Y={y_mean:.0f})")
                    
                    file_bounds_info.append({
                        'file': os.path.basename(laz_file),
                        'crs': point_crs,
                        'x_range': (x_min, x_max),
                        'y_range': (y_min, y_max),
                        'point_count': len(las.x)
                    })
                
            except Exception as e:
                if config.VERBOSE:
                    print(f"    Warning: Could not read {laz_file} for diagnostics: {e}")
                continue
    
    # Second pass: Load with bounds filtering
    for laz_file in laz_files:
        if config.VERBOSE:
            print(f"  Loading {os.path.basename(laz_file)}...")
        
        try:
            # Try to use lazrs backend explicitly
            try:
                import lazrs
                if hasattr(laspy, 'LazBackend'):
                    las = laspy.read(laz_file, laz_backend=laspy.LazBackend.Lazrs)
                else:
                    las = laspy.read(laz_file, laz_backend='lazrs')
            except (ImportError, AttributeError, TypeError, ValueError) as e:
                if config.VERBOSE:
                    print(f"    Note: Using default backend (error: {e})")
                las = laspy.read(laz_file)
            
            # Detect CRS from coordinate ranges (if not already set)
            if point_crs is None:
                x_sample = las.x[:1000] if len(las.x) > 1000 else las.x
                y_sample = las.y[:1000] if len(las.y) > 1000 else las.y
                
                x_mean = np.mean(x_sample)
                y_mean = np.mean(y_sample)
                
                if 900000 < x_mean < 1100000 and 100000 < y_mean < 400000:
                    point_crs = CRS.from_epsg(2263)
                    if config.VERBOSE:
                        print(f"    Detected CRS: EPSG:2263 (State Plane, feet) from coordinates")
                elif 400000 < x_mean < 700000 and 4000000 < y_mean < 5000000:
                    point_crs = CRS.from_epsg(3178)
                    if config.VERBOSE:
                        print(f"    Detected CRS: EPSG:3178 (Albers, meters) from coordinates")
                else:
                    point_crs = default_lidar_crs
                    if config.VERBOSE:
                        print(f"    Using default CRS: EPSG:2263 (coordinates: X={x_mean:.0f}, Y={y_mean:.0f})")
            
            # Extract points
            x = las.x
            y = las.y
            z = las.z
            
            # Filter by bounds if provided
            if bounds is not None and bounds_crs is not None:
                # Convert bounds to regular floats and ensure min < max
                minx, miny, maxx, maxy = [float(b) for b in bounds]
                
                # Ensure min < max (handle potential coordinate system issues)
                if minx > maxx:
                    minx, maxx = maxx, minx
                if miny > maxy:
                    miny, maxy = maxy, miny
                
                # Transform bounds to point CRS if needed
                if bounds_crs != point_crs:
                    if config.VERBOSE:
                        print(f"    Transforming bounds from {bounds_crs} to {point_crs}...")
                    from rasterio.warp import transform_bounds
                    try:
                        minx, miny, maxx, maxy = transform_bounds(
                            bounds_crs, point_crs,
                            minx, miny, maxx, maxy
                        )
                        # Ensure min < max after transformation
                        if minx > maxx:
                            minx, maxx = maxx, minx
                        if miny > maxy:
                            miny, maxy = maxy, miny
                        if config.VERBOSE:
                            print(f"    Transformed bounds: ({minx:.2f}, {miny:.2f}, {maxx:.2f}, {maxy:.2f})")
                    except Exception as e:
                        if config.VERBOSE:
                            print(f"    Warning: Bounds transformation failed: {e}")
                        # Continue without bounds filtering
                        bounds = None
                
                if bounds is not None:
                    # Filter points with a small buffer to account for rounding
                    buffer = 0.01  # Small buffer in CRS units
                    mask = (x >= (minx - buffer)) & (x <= (maxx + buffer)) & \
                           (y >= (miny - buffer)) & (y <= (maxy + buffer))
                    x = x[mask]
                    y = y[mask]
                    z = z[mask]
                    
                    if config.VERBOSE:
                        print(f"    Points after bounds filtering: {len(x)}")
                        if len(x) == 0:
                            # Show diagnostic info
                            x_range = (np.min(las.x), np.max(las.x))
                            y_range = (np.min(las.y), np.max(las.y))
                            print(f"      File X range: {x_range[0]:.2f} to {x_range[1]:.2f}")
                            print(f"      File Y range: {y_range[0]:.2f} to {y_range[1]:.2f}")
                            print(f"      Target X range: {minx:.2f} to {maxx:.2f}")
                            print(f"      Target Y range: {miny:.2f} to {maxy:.2f}")
            
            # Stack into array
            if len(x) > 0:
                points = np.column_stack([x, y, z])
                all_points.append(points)
                
                if config.VERBOSE:
                    print(f"    Loaded {len(points)} points")
        
        except Exception as e:
            if config.VERBOSE:
                print(f"    Warning: Could not load {laz_file}: {e}")
                import traceback
                if config.VERBOSE:
                    traceback.print_exc()
            continue
    
    if not all_points:
        error_msg = "No points loaded from LiDAR files"
        if bounds is not None and diagnostic_mode:
            error_msg += "\n\nDiagnostic information:"
            for info in file_bounds_info:
                error_msg += f"\n  {info['file']}: {info['point_count']} points, "
                error_msg += f"X: {info['x_range'][0]:.2f}-{info['x_range'][1]:.2f}, "
                error_msg += f"Y: {info['y_range'][0]:.2f}-{info['y_range'][1]:.2f}"
        raise ValueError(error_msg)
    
    # Concatenate all points
    all_points = np.vstack(all_points)
    
    if config.VERBOSE:
        print(f"  Total points loaded: {len(all_points)}")
        print(f"  Point CRS: {point_crs}")
        if len(all_points) > 0:
            print(f"  Point X range: {np.min(all_points[:, 0]):.2f} to {np.max(all_points[:, 0]):.2f}")
            print(f"  Point Y range: {np.min(all_points[:, 1]):.2f} to {np.max(all_points[:, 1]):.2f}")
    
    return all_points, point_crs


def rasterize_lidar_points(points, target_transform, target_shape, target_crs, 
                           point_crs=None, method='mean', radius=None):
    """
    Rasterize LiDAR points into a DEM.
    
    Parameters:
    -----------
    points : ndarray
        Array of shape (n_points, 3) with columns [x, y, z]
    target_transform : Affine
        Target raster transform
    target_shape : tuple
        (height, width) of target raster
    target_crs : CRS
        Target CRS
    point_crs : CRS, optional
        CRS of input points (will transform if different from target_crs)
    method : str
        Method for aggregating points: 'mean', 'min', 'max', 'median'
    radius : float, optional
        Search radius for interpolation (in CRS units)
        
    Returns:
    --------
    dem : ndarray
        Digital Elevation Model raster (height, width)
    """
    if config.VERBOSE:
        print(f"\nRasterizing {len(points)} points to {target_shape}...")
    
    height, width = target_shape
    dem = np.full((height, width), np.nan, dtype=np.float32)
    
    # Extract coordinates
    x_coords = points[:, 0]
    y_coords = points[:, 1]
    z_values = points[:, 2]
    
    # Transform points to target CRS if needed
    if point_crs is not None and point_crs != target_crs:
        if config.VERBOSE:
            print(f"  Transforming points from {point_crs} to {target_crs}...")
        from rasterio.warp import transform
        x_coords, y_coords = transform(
            point_crs, target_crs,
            x_coords, y_coords
        )
    
    # Convert point coordinates to pixel indices using inverse transform
    # Using rowcol function from rasterio.transform
    from rasterio.transform import rowcol
    rows, cols = rowcol(target_transform, x_coords, y_coords)
    
    # Filter points within raster bounds
    valid = (rows >= 0) & (rows < height) & (cols >= 0) & (cols < width)
    rows = rows[valid].astype(int)
    cols = cols[valid].astype(int)
    z_values = z_values[valid]
    
    if len(z_values) == 0:
        # Provide diagnostic information
        error_msg = "No points fall within target raster bounds\n"
        error_msg += f"  Target raster shape: {target_shape} (height x width)\n"
        error_msg += f"  Target CRS: {target_crs}\n"
        if len(points) > 0:
            error_msg += f"  Total points loaded: {len(points)}\n"
            error_msg += f"  Point X range: {np.min(points[:, 0]):.2f} to {np.max(points[:, 0]):.2f}\n"
            error_msg += f"  Point Y range: {np.min(points[:, 1]):.2f} to {np.max(points[:, 1]):.2f}\n"
            if point_crs is not None:
                error_msg += f"  Point CRS: {point_crs}\n"
            
            # Calculate raster bounds for comparison
            from rasterio.transform import xy
            raster_corners = [
                xy(target_transform, 0, 0),
                xy(target_transform, target_shape[1], 0),
                xy(target_transform, 0, target_shape[0]),
                xy(target_transform, target_shape[1], target_shape[0])
            ]
            raster_x = [c[0] for c in raster_corners]
            raster_y = [c[1] for c in raster_corners]
            error_msg += f"  Raster X range: {min(raster_x):.2f} to {max(raster_x):.2f}\n"
            error_msg += f"  Raster Y range: {min(raster_y):.2f} to {max(raster_y):.2f}\n"
        raise ValueError(error_msg)
    
    if config.VERBOSE:
        print(f"  {len(z_values)} points within raster bounds")
    
    # Rasterize using specified method
    if method == 'mean':
        # Use bincount for efficient aggregation
        indices = rows * width + cols
        # Count occurrences and sum values
        counts = np.bincount(indices, minlength=height * width)
        sums = np.bincount(indices, weights=z_values, minlength=height * width)
        # Calculate mean (avoid division by zero)
        valid_indices = counts > 0
        dem_flat = np.full(height * width, np.nan, dtype=np.float32)
        dem_flat[valid_indices] = sums[valid_indices] / counts[valid_indices]
        dem = dem_flat.reshape(height, width)
    
    elif method == 'min':
        # Use minimum z per pixel
        indices = rows * width + cols
        unique_indices = np.unique(indices)
        for idx in unique_indices:
            mask = indices == idx
            pixel_row = idx // width
            pixel_col = idx % width
            dem[pixel_row, pixel_col] = np.min(z_values[mask])
    
    elif method == 'max':
        # Use maximum z per pixel
        indices = rows * width + cols
        unique_indices = np.unique(indices)
        for idx in unique_indices:
            mask = indices == idx
            pixel_row = idx // width
            pixel_col = idx % width
            dem[pixel_row, pixel_col] = np.max(z_values[mask])
    
    else:
        raise ValueError(f"Unknown rasterization method: {method}")
    
    # Fill NaN values with interpolation if needed
    if np.isnan(dem).any():
        try:
            from scipy.interpolate import griddata
            valid_mask = ~np.isnan(dem)
            if valid_mask.sum() > 0:
                rows_valid, cols_valid = np.where(valid_mask)
                z_valid = dem[valid_mask]
                rows_all, cols_all = np.meshgrid(np.arange(height), np.arange(width), indexing='ij')
                dem_filled = griddata(
                    (rows_valid, cols_valid), z_valid,
                    (rows_all, cols_all),
                    method='nearest',
                    fill_value=np.nan
                )
                dem = np.where(np.isnan(dem), dem_filled, dem)
        except ImportError:
            if config.VERBOSE:
                print("  Warning: scipy not available for interpolation")
    
    if config.VERBOSE:
        valid_count = (~np.isnan(dem)).sum()
        print(f"  DEM created: shape {dem.shape}, valid pixels: {valid_count}")
        if valid_count > 0:
            print(f"  Elevation range: {np.nanmin(dem):.2f} to {np.nanmax(dem):.2f} m")
    
    return dem


def calculate_ground_level_dtm(dem, window_size=50):
    """
    Estimate ground level (DTM) by finding minimum heights in large windows.
    This helps distinguish rooftops (high above ground) from sidewalks (at ground level).
    
    Parameters:
    -----------
    dem : ndarray
        Digital Elevation Model (DSM - includes buildings)
    window_size : int
        Size of window for finding ground level (default: 50x50)
        
    Returns:
    --------
    dtm : ndarray
        Digital Terrain Model (ground level estimate)
    """
    from scipy.ndimage import minimum_filter
    
    # Use minimum filter to estimate ground level (lowest points in neighborhood)
    # This assumes ground is lower than buildings
    dtm = minimum_filter(dem, size=window_size, mode='constant', cval=np.nan)
    
    # Fill NaN with original DEM values
    dtm = np.where(np.isnan(dtm), dem, dtm)
    
    return dtm


def calculate_relative_height_offset(dem, window_size=5, threshold_cm=10, use_ground_level=True):
    """
    Calculate height offset for each pixel.
    
    Two modes:
    1. If use_ground_level=True: Compare to estimated ground level (DTM) to distinguish
       rooftops (high above ground) from sidewalks (at ground level)
    2. If use_ground_level=False: Compare to local neighborhood (original method)
    
    Parameters:
    -----------
    dem : ndarray
        Digital Elevation Model (height, width)
    window_size : int
        Size of window for calculating surrounding height (default: 5x5)
    threshold_cm : float
        Height difference threshold in centimeters (default: 10)
    use_ground_level : bool
        If True, compare to ground level (DTM) instead of local mean
        
    Returns:
    --------
    height_offset : ndarray
        Height offset in meters (height, width)
    is_similar_height : ndarray
        Boolean mask: True if height difference < threshold
    """
    if config.VERBOSE:
        mode_str = "ground level (DTM)" if use_ground_level else "local neighborhood"
        print(f"\nCalculating height offsets (threshold: {threshold_cm} cm, mode: {mode_str})...")
    
    # Convert threshold to meters
    threshold_m = threshold_cm / 100.0
    
    # Create a mask for valid (non-NaN) pixels
    valid_mask = ~np.isnan(dem)
    
    if use_ground_level:
        # Method 1: Compare to estimated ground level (DTM)
        # This better distinguishes rooftops (high above ground) from sidewalks (at ground level)
        dtm = calculate_ground_level_dtm(dem, window_size=50)
        reference_height = dtm
        height_offset = dem - dtm  # Height above ground level
        
        if config.VERBOSE:
            print(f"  Ground level (DTM) range: {np.nanmin(dtm):.2f} to {np.nanmax(dtm):.2f} m")
            print(f"  Height above ground range: {np.nanmin(height_offset[valid_mask]):.2f} to {np.nanmax(height_offset[valid_mask]):.2f} m")
            
            # Count pixels at different height levels
            at_ground = (height_offset[valid_mask] < 0.5).sum()  # < 50cm above ground (sidewalks)
            elevated = (height_offset[valid_mask] >= 2.0).sum()  # >= 2m above ground (likely rooftops)
            print(f"  Pixels at ground level (<0.5m): {at_ground} ({100*at_ground/valid_mask.sum():.1f}%)")
            print(f"  Pixels elevated (>=2m): {elevated} ({100*elevated/valid_mask.sum():.1f}%)")
    else:
        # Method 2: Compare to local neighborhood (original method)
        from scipy.ndimage import uniform_filter
        
        # Fill NaN with 0 temporarily for filtering
        dem_filled = np.where(valid_mask, dem, 0.0)
        
        # Calculate sum of valid values in each window
        sum_height = uniform_filter(dem_filled, size=window_size, mode='constant', cval=0.0) * (window_size ** 2)
        
        # Calculate count of valid pixels in each window
        valid_count = uniform_filter(valid_mask.astype(float), size=window_size, mode='constant', cval=0.0) * (window_size ** 2)
        
        # Calculate mean by dividing sum by count (only where count > 0)
        reference_height = np.full_like(dem, np.nan, dtype=np.float32)
        has_enough_data = valid_count > 0
        reference_height[has_enough_data] = sum_height[has_enough_data] / valid_count[has_enough_data]
        
        # Calculate offset (pixel height - surrounding mean)
        height_offset = dem - reference_height
    
    # Mask for similar height (within threshold)
    # For ground level mode: keep pixels close to ground (sidewalks)
    # For local mode: keep pixels similar to neighbors
    valid_for_comparison = valid_mask & ~np.isnan(reference_height)
    is_similar_height = np.zeros_like(dem, dtype=bool)
    
    if use_ground_level:
        # Keep pixels at or near ground level (sidewalks, not rooftops)
        # Strategy: Only exclude very high rooftops (>5m above ground), be very lenient with everything else
        # This allows sidewalks with elevation variation (curbs, ramps, slopes) while removing rooftops
        rooftop_exclusion_threshold = getattr(config, 'ROOFTOP_EXCLUSION_THRESHOLD_M', 5.0)  # meters above ground - increased to be less aggressive
        use_local_variation = getattr(config, 'USE_LOCAL_VARIATION_FILTER', False)  # Can disable for less aggressive filtering
        
        # Primary filter: Exclude only very high rooftops (high above ground)
        # This is the main filter - be lenient to preserve sidewalks
        at_ground_level = height_offset[valid_for_comparison] < rooftop_exclusion_threshold
        
        if use_local_variation:
            # Optional secondary filter: Check local variation (can be disabled)
            from scipy.ndimage import uniform_filter
            dem_filled = np.where(valid_mask, dem, 0.0)
            local_mean = uniform_filter(dem_filled, size=window_size, mode='constant', cval=0.0)
            local_variation = np.abs(dem[valid_for_comparison] - local_mean[valid_for_comparison])
            local_variation_ok = local_variation < (threshold_m * 3.0)  # Very lenient: 3x threshold
            # Keep pixels that are at ground level AND have reasonable local variation
            is_similar_height[valid_for_comparison] = at_ground_level & local_variation_ok
        else:
            # Only use rooftop exclusion - no local variation check (less aggressive)
            # This preserves all ground-level pixels, only excluding very high rooftops
            is_similar_height[valid_for_comparison] = at_ground_level
    else:
        # Original method: keep pixels similar to local neighborhood
        is_similar_height[valid_for_comparison] = np.abs(height_offset[valid_for_comparison]) < threshold_m
    
    # Set NaN pixels to False (not similar height)
    is_similar_height[~valid_for_comparison] = False
    
    if config.VERBOSE:
        similar_count = is_similar_height[valid_for_comparison].sum()
        total_valid = valid_for_comparison.sum()
        if total_valid > 0:
            print(f"  Pixels with similar height (< {threshold_cm} cm): {similar_count} / {total_valid} ({100*similar_count/total_valid:.1f}%)")
            if use_ground_level:
                print(f"  (Pixels at/near ground level, excluding rooftops)")
            print(f"  Height offset range: {np.nanmin(height_offset[valid_for_comparison]):.3f} to {np.nanmax(height_offset[valid_for_comparison]):.3f} m")
        else:
            print(f"  WARNING: No valid pixels for height comparison!")
            print(f"  Valid DEM pixels: {valid_mask.sum()}")
            print(f"  Valid reference height pixels: {(~np.isnan(reference_height)).sum()}")
    
    return height_offset, is_similar_height


def load_and_align_lidar_elevation(lidar_dir, hyperspectral_profile, hyperspectral_transform,
                                   hyperspectral_bounds=None, rasterize_method='mean',
                                   height_threshold_cm=10):
    """
    Load LiDAR elevation data and align with hyperspectral data.
    
    Parameters:
    -----------
    lidar_dir : str
        Path to directory containing .laz files
    hyperspectral_profile : dict
        Hyperspectral rasterio profile
    hyperspectral_transform : Affine
        Hyperspectral transform
    hyperspectral_bounds : tuple, optional
        (minx, miny, maxx, maxy) bounds of hyperspectral window
    rasterize_method : str
        Method for rasterizing points: 'mean', 'min', 'max'
    height_threshold_cm : float
        Height difference threshold in cm for filtering
        
    Returns:
    --------
    dem : ndarray
        Digital Elevation Model aligned with hyperspectral data
    height_offset : ndarray
        Relative height offset map
    is_similar_height : ndarray
        Boolean mask for similar height pixels
    """
    if config.VERBOSE:
        print(f"\n{'='*60}")
        print("LOADING LIDAR ELEVATION DATA")
        print(f"{'='*60}")
        print(f"LiDAR directory: {lidar_dir}")
    
    # Find LiDAR files
    laz_files = find_lidar_files(lidar_dir)
    
    # Calculate hyperspectral bounds if not provided
    if hyperspectral_bounds is None:
        hs_height = hyperspectral_profile['height']
        hs_width = hyperspectral_profile['width']
        
        # Get all four corners to ensure correct min/max calculation
        # This handles transforms with different orientations
        corners = [
            xy(hyperspectral_transform, 0, 0),           # Top-left
            xy(hyperspectral_transform, hs_width, 0),      # Top-right
            xy(hyperspectral_transform, 0, hs_height),     # Bottom-left
            xy(hyperspectral_transform, hs_width, hs_height)  # Bottom-right
        ]
        
        x_coords = [c[0] for c in corners]
        y_coords = [c[1] for c in corners]
        
        minx = float(min(x_coords))
        maxx = float(max(x_coords))
        miny = float(min(y_coords))
        maxy = float(max(y_coords))
        
        hyperspectral_bounds = (minx, miny, maxx, maxy)
    else:
        # Ensure bounds are regular floats, not numpy types
        hyperspectral_bounds = tuple(float(b) for b in hyperspectral_bounds)
        minx, miny, maxx, maxy = hyperspectral_bounds
        # Ensure min < max
        if minx > maxx:
            minx, maxx = maxx, minx
        if miny > maxy:
            miny, maxy = maxy, miny
        hyperspectral_bounds = (minx, miny, maxx, maxy)
    
    if config.VERBOSE:
        print(f"Hyperspectral bounds: {hyperspectral_bounds}")
        print(f"  (minx={hyperspectral_bounds[0]:.2f}, miny={hyperspectral_bounds[1]:.2f}, "
              f"maxx={hyperspectral_bounds[2]:.2f}, maxy={hyperspectral_bounds[3]:.2f})")
    
    # Load LiDAR points with diagnostic mode enabled
    # Note: bounds are in hyperspectral CRS, but points might be in different CRS
    hs_crs = hyperspectral_profile['crs']
    
    # Try with original bounds first
    points = None
    point_crs = None
    expand_factor = getattr(config, 'LIDAR_BOUNDS_EXPAND_FACTOR', 1.0)
    max_expansions = getattr(config, 'LIDAR_MAX_BOUNDS_EXPANSIONS', 2)
    
    for expansion_attempt in range(max_expansions + 1):
        try:
            current_bounds = hyperspectral_bounds
            if expansion_attempt > 0:
                # Expand bounds by a factor
                minx, miny, maxx, maxy = hyperspectral_bounds
                center_x = (minx + maxx) / 2.0
                center_y = (miny + maxy) / 2.0
                width = maxx - minx
                height = maxy - miny
                
                # Expand by expansion_factor each time
                expansion = expand_factor ** expansion_attempt
                new_width = width * expansion
                new_height = height * expansion
                
                minx = center_x - new_width / 2.0
                maxx = center_x + new_width / 2.0
                miny = center_y - new_height / 2.0
                maxy = center_y + new_height / 2.0
                
                current_bounds = (minx, miny, maxx, maxy)
                if config.VERBOSE and expansion_attempt == 1:
                    print(f"  Expanding search bounds by {expand_factor}x (attempt {expansion_attempt + 1})...")
            
            points, point_crs = load_lidar_points(
                laz_files, 
                bounds=current_bounds, 
                bounds_crs=hs_crs,
                diagnostic_mode=(expansion_attempt == 0)  # Only diagnostic on first attempt
            )
            break  # Success!
            
        except ValueError as e:
            if expansion_attempt == 0:
                # First attempt failed - try loading without bounds for diagnostics
                if config.VERBOSE:
                    print("\n  Attempting to load points without bounds for diagnostics...")
                try:
                    diagnostic_points, diagnostic_crs = load_lidar_points(
                        laz_files,
                        bounds=None,
                        bounds_crs=None,
                        diagnostic_mode=True
                    )
                    if len(diagnostic_points) > 0:
                        print(f"  Found {len(diagnostic_points)} points in files (without bounds filter)")
                        print(f"  Point X range: {np.min(diagnostic_points[:, 0]):.2f} to {np.max(diagnostic_points[:, 0]):.2f}")
                        print(f"  Point Y range: {np.min(diagnostic_points[:, 1]):.2f} to {np.max(diagnostic_points[:, 1]):.2f}")
                        print(f"  Point CRS: {diagnostic_crs}")
                        print(f"  Hyperspectral bounds (in {hs_crs}): {hyperspectral_bounds}")
                        
                        # Try to transform hyperspectral bounds to point CRS for comparison
                        try:
                            from rasterio.warp import transform_bounds
                            hs_bounds_in_point_crs = transform_bounds(
                                hs_crs, diagnostic_crs,
                                *hyperspectral_bounds
                            )
                            print(f"  Hyperspectral bounds (in {diagnostic_crs}): {hs_bounds_in_point_crs}")
                            
                            # Check if there's any overlap
                            hs_minx, hs_miny, hs_maxx, hs_maxy = hs_bounds_in_point_crs
                            pt_minx, pt_miny = np.min(diagnostic_points[:, 0]), np.min(diagnostic_points[:, 1])
                            pt_maxx, pt_maxy = np.max(diagnostic_points[:, 0]), np.max(diagnostic_points[:, 1])
                            
                            if (hs_maxx < pt_minx or hs_minx > pt_maxx or 
                                hs_maxy < pt_miny or hs_miny > pt_maxy):
                                print("  WARNING: No overlap detected between hyperspectral bounds and LiDAR coverage!")
                            else:
                                print("  Overlap detected - bounds may need adjustment")
                        except Exception as te:
                            if config.VERBOSE:
                                print(f"  Could not transform bounds for comparison: {te}")
                except:
                    pass
            
            # If this was the last attempt, raise the error
            if expansion_attempt >= max_expansions:
                raise e
            # Otherwise, continue to next expansion attempt
    
    # Rasterize points
    hs_height = hyperspectral_profile['height']
    hs_width = hyperspectral_profile['width']
    dem = rasterize_lidar_points(
        points,
        hyperspectral_transform,
        (hs_height, hs_width),
        hs_crs,
        point_crs=point_crs,
        method=rasterize_method
    )
    
    # Calculate relative height offsets
    # Use ground level mode to better distinguish rooftops from sidewalks
    use_ground_level = getattr(config, 'USE_GROUND_LEVEL_FILTERING', True)
    height_offset, is_similar_height = calculate_relative_height_offset(
        dem,
        window_size=5,
        threshold_cm=height_threshold_cm,
        use_ground_level=use_ground_level
    )
    
    if config.VERBOSE:
        print(f"\n[OK] LiDAR elevation data loaded and aligned")
        print(f"  DEM shape: {dem.shape}")
        valid_count = (~np.isnan(dem)).sum()
        print(f"  Valid pixels: {valid_count}")
    
    return dem, height_offset, is_similar_height

