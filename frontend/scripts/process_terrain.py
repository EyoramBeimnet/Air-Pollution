"""Resample USGS 3DEP 1/3 arc-second DEMs (USGS_13_n41w074.tif, USGS_13_n41w075.tif) to a 1024x1024 heightmap.

Usage: python process_terrain.py DEM_DIR OUT_DIR   ->  height.bin (int16 decimetres, row 0 = north)
"""
import numpy as np, rasterio, os, sys
from rasterio.warp import reproject, Resampling
from rasterio.transform import from_bounds
LON0, LAT0 = -73.975, 40.703; COS = np.cos(np.radians(LAT0)); K = 60.0
X0, X1, Y0, Y1 = -13.2, 12.8, -12.7, 13.3; N = 1024
src_dir, out = sys.argv[1], sys.argv[2]; os.makedirs(out, exist_ok=True)
tr = from_bounds(LON0 + X0/(COS*K), LAT0 + Y0/K, LON0 + X1/(COS*K), LAT0 + Y1/K, N, N)
dst = np.full((N, N), np.nan, np.float32)
for fn in ['USGS_13_n41w074.tif', 'USGS_13_n41w075.tif']:
    with rasterio.open(os.path.join(src_dir, fn)) as src:
        part = np.full((N, N), np.nan, np.float32)
        reproject(rasterio.band(src, 1), part, dst_transform=tr, dst_crs='EPSG:4269', resampling=Resampling.average, dst_nodata=np.nan)
        m = np.isnan(dst) & ~np.isnan(part); dst[m] = part[m]
np.clip(np.round(np.nan_to_num(dst) * 10), -32768, 32767).astype('<i2').tofile(os.path.join(out, 'height.bin'))
