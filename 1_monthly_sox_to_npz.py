import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt
from scipy.spatial import cKDTree
from matplotlib.colors import LogNorm, Normalize
import cartopy.crs as ccrs
import cartopy.feature as cfeature

NC_DIR = '/home/chenyiqi/260320_ship_emission/ship_emission_cams/CAMS-GLOB-ANT_Glb_0.1x0.1_anthro_so2_v6.2_monthly'
NC_FILENAME_TEMPLATE = 'CAMS-GLOB-ANT_Glb_0.1x0.1_anthro_so2_v6.2_monthly_{year}.nc'
SUM_VAR = 'sum'
OUT_DIR = '/home/chenyiqi/260320_ship_emission'
YEAR_START = 2000
YEAR_END = 2022
LSMASK_PATH = '/data/chenyiqi/251007_tropic/landsea.nc'

# ====================== 【新增开关】 ======================
SAVE_NPZ = False                 # 是否保存 processed npz → True/False
# ==========================================================

# Set to 'log' for logarithmic color scaling, or 'linear' for linear color scaling.
COLOR_SCALE_MODE = 'log'

# Shared color settings for all annual maps.
SHARED_CMAP = 'viridis'
SHARED_VMIN = None
SHARED_VMAX = None
LAT_EDGES = np.arange(-90, 91, 1)
LON_EDGES = np.arange(-180, 181, 1)
OCEAN_MASK_1DEG = None


def _latlon_to_unit_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
    lat_rad = np.deg2rad(lat_deg.astype(float))
    lon_rad = np.deg2rad(lon_deg.astype(float))
    clat = np.cos(lat_rad)
    x = clat * np.cos(lon_rad)
    y = clat * np.sin(lon_rad)
    z = np.sin(lat_rad)
    return np.column_stack([x, y, z])


def _build_ocean_mask_1deg():
    """Build 1x1 degree ocean mask from LSMASK and remove coastal ocean cells."""
    with nc.Dataset(LSMASK_PATH, 'r') as ds:
        lat_lsm = ds.variables['lat'][:].astype(float)
        lon_lsm = ds.variables['lon'][:].astype(float)
        lsmask = ds.variables['LSMASK'][:].astype(bool)

    lon_lsm = ((lon_lsm + 180.0) % 360.0) - 180.0
    lon_grid, lat_grid = np.meshgrid(lon_lsm, lat_lsm)
    land_mask = np.flipud(lsmask.astype(bool))
    lat_grid = np.flipud(lat_grid)
    lon_grid = np.flipud(lon_grid)

    half_width = land_mask.shape[1] // 2
    if half_width > 0:
        land_mask = np.hstack([land_mask[:, half_width:], land_mask[:, :half_width]])
        lon_grid = np.hstack([lon_grid[:, half_width:], lon_grid[:, :half_width]])
        lat_grid = np.hstack([lat_grid[:, half_width:], lat_grid[:, :half_width]])

    ocean_native = ~land_mask

    lat_native = lat_grid[:, 0]
    lon_native = lon_grid[0, :]
    lat_centers = (LAT_EDGES[:-1] + LAT_EDGES[1:]) / 2.0
    lon_centers = (LON_EDGES[:-1] + LON_EDGES[1:]) / 2.0

    lat_idx = np.abs(lat_native[:, None] - lat_centers[None, :]).argmin(axis=0)
    lon_dist = np.abs(((lon_native[:, None] - lon_centers[None, :]) + 180.0) % 360.0 - 180.0)
    lon_idx = lon_dist.argmin(axis=0)

    ocean_mask_1deg = ocean_native[np.ix_(lat_idx, lon_idx)]

    lon2d, lat2d = np.meshgrid(lon_centers, lat_centers)
    land_mask_1deg = ~ocean_mask_1deg
    land_lat = lat2d[land_mask_1deg]
    land_lon = lon2d[land_mask_1deg]

    if land_lat.size > 0:
        land_xyz = _latlon_to_unit_xyz(land_lat, land_lon)
        tree = cKDTree(land_xyz)
        ocean_lat = lat2d[ocean_mask_1deg]
        ocean_lon = lon2d[ocean_mask_1deg]
        ocean_xyz = _latlon_to_unit_xyz(ocean_lat, ocean_lon)
        chord_dist, _ = tree.query(ocean_xyz, k=1)
        clipped = np.clip(chord_dist / 2.0, 0.0, 1.0)
        angle_rad = 2.0 * np.arcsin(clipped)
        dist_km = 6371.0 * angle_rad

        keep_ocean = np.zeros_like(ocean_mask_1deg, dtype=bool)
        keep_ocean[ocean_mask_1deg] = dist_km >= 500.0
        ocean_mask_1deg = keep_ocean

    return ocean_mask_1deg


def _fill_missing_ocean_sox_in_trop_midlat(sox_grid: np.ndarray, ocean_mask: np.ndarray) -> np.ndarray:
    """Fill missing ocean SOx with 0 for grid-cell centers between 60N and 60S."""
    filled = sox_grid.copy()
    lat_centers = (LAT_EDGES[:-1] + LAT_EDGES[1:]) / 2.0
    lat_band_mask = (lat_centers >= -60.0) & (lat_centers <= 60.0)
    lat_band_2d = lat_band_mask[:, None]
    fill_mask = ocean_mask & lat_band_2d & ~np.isfinite(filled)
    filled[fill_mask] = 0.0
    return filled


def _get_ocean_mask_1deg():
    global OCEAN_MASK_1DEG
    if OCEAN_MASK_1DEG is None:
        OCEAN_MASK_1DEG = _build_ocean_mask_1deg()
    return OCEAN_MASK_1DEG


def _aggregate_to_1deg_grid(annual_mean_01deg: np.ndarray, lat: np.ndarray, lon: np.ndarray) -> np.ndarray:
    """Aggregate 0.1° annual mean data into 1° grid cells."""
    if annual_mean_01deg.shape != (len(lat), len(lon)):
        raise ValueError('Input annual_mean_01deg shape does not match lat/lon dimensions.')

    lon_2d, lat_2d = np.meshgrid(lon, lat)
    lon_flat = lon_2d.ravel()
    lat_flat = lat_2d.ravel()
    value_flat = annual_mean_01deg.ravel()

    lon_flat = ((lon_flat + 180.0) % 360.0) - 180.0
    lon_bin = np.floor(lon_flat).astype(int)
    lat_bin = np.floor(lat_flat).astype(int)
    lon_bin = lon_bin.clip(-180, 179)
    lat_bin = lat_bin.clip(-90, 89)

    lat_idx = lat_bin - (-90)
    lon_idx = lon_bin - (-180)

    sum_grid = np.zeros((len(LAT_EDGES) - 1, len(LON_EDGES) - 1), dtype=np.float64)
    count_grid = np.zeros_like(sum_grid, dtype=np.int32)

    finite_mask = np.isfinite(value_flat)
    np.add.at(sum_grid, (lat_idx[finite_mask], lon_idx[finite_mask]), value_flat[finite_mask])
    np.add.at(count_grid, (lat_idx[finite_mask], lon_idx[finite_mask]), 1)

    grid = np.full_like(sum_grid, np.nan, dtype=np.float64)
    valid = count_grid > 0
    grid[valid] = sum_grid[valid]
    return grid


def load_monthly_1deg(year: int) -> np.ndarray:
    """Load monthly 0.1deg data from NetCDF and aggregate each month to 1deg.

    Returns an array shape (12, nlat_1deg, nlon_1deg).
    """
    nc_path = os.path.join(NC_DIR, NC_FILENAME_TEMPLATE.format(year=year))
    if not os.path.exists(nc_path):
        raise FileNotFoundError(f'NetCDF file not found: {nc_path}')

    with nc.Dataset(nc_path, 'r') as ds:
        if SUM_VAR not in ds.variables:
            raise KeyError(f"Variable '{SUM_VAR}' not found in {nc_path}")

        monthly_data = ds.variables[SUM_VAR][:].astype(np.float64)
        if monthly_data.ndim != 3 or monthly_data.shape[0] != 12:
            raise ValueError(f'Expected 12 monthly time steps in {nc_path}, got shape {monthly_data.shape}')

        lat = ds.variables['lat'][:].astype(np.float64)
        lon = ds.variables['lon'][:].astype(np.float64)

    monthly_grids = []
    for m in range(monthly_data.shape[0]):
        monthly_grids.append(_aggregate_to_1deg_grid(monthly_data[m], lat, lon))
    return np.stack(monthly_grids, axis=0)


def _build_shared_norm(all_grids: list[np.ndarray]):
    mode = COLOR_SCALE_MODE.lower().strip()
    if mode not in {'log', 'linear'}:
        raise ValueError("COLOR_SCALE_MODE must be either 'log' or 'linear'.")

    if SHARED_VMIN is not None and SHARED_VMAX is not None:
        vmin = SHARED_VMIN
        vmax = SHARED_VMAX
    else:
        values = np.concatenate([grid[np.isfinite(grid)] for grid in all_grids])
        if mode == 'log':
            values = values[values > 0.0]
        if values.size == 0:
            raise ValueError('No finite positive values are available for color scaling.')
        vmin = float(np.nanmin(values))
        vmax = float(np.nanmax(values))

    if mode == 'log':
        return LogNorm(vmin=vmin, vmax=vmax)
    return Normalize(vmin=vmin, vmax=vmax)


def plot_year_mean(year: int, monthly_grids: np.ndarray, norm) -> None:
    out_path = os.path.join(OUT_DIR, 'figs', f'sox_monthly_mean_{year}.png')
    plot_grid = np.nanmean(monthly_grids, axis=0)
    if COLOR_SCALE_MODE.lower().strip() == 'log':
        plot_grid = np.where(plot_grid > 0.0, plot_grid, np.nan)

    fig = plt.figure(figsize=(12, 6), dpi=300)
    ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
    hb = ax.pcolormesh(
        LON_EDGES,
        LAT_EDGES,
        plot_grid,
        cmap=SHARED_CMAP,
        norm=norm,
        shading='auto',
        transform=ccrs.PlateCarree(),
    )
    ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none', zorder=2)
    ax.coastlines(resolution='110m', linewidth=0.6, color='black', zorder=3)
    gl = ax.gridlines(draw_labels=True, linewidth=0.4, linestyle='--', alpha=0.35)
    gl.top_labels = False
    gl.right_labels = False
    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_title(f'SOx Monthly Mean {year}')
    ax.set_xlabel('Longitude')
    ax.set_ylabel('Latitude')

    cbar = fig.colorbar(hb, ax=ax, extend='both', pad=0.02)
    cbar.set_label('SOx per 1deg x 1deg grid')
    cbar.ax.tick_params(labelsize=8)
    fig.savefig(out_path, bbox_inches='tight')
    plt.close(fig)
    print(f'Saved figure: {out_path}')


# ====================== 【新增：全球平均年均 SO2 年际变化曲线】 ======================
def plot_global_annual_mean_timeseries(years, annual_mean_values):
    out_path = os.path.join(OUT_DIR, 'figs', 'global_annual_mean_so2_timeseries.png')
    plt.figure(figsize=(8, 5), dpi=300)
    
    # 画主线
    plt.plot(years, annual_mean_values, 'o-', color='#2E86AB', linewidth=2.5, markersize=6)
    
    # ---------------- 竖线：2005 和 2019 ----------------
    plt.axvline(x=2005, color='red', linestyle='--', linewidth=1.5, alpha=0.8, label='2005')
    plt.axvline(x=2019, color='blue', linestyle='--', linewidth=1.5, alpha=0.8, label='2019')
    
    # 网格
    plt.grid(True, alpha=0.3, linestyle='--')
    
    # 字体全部加大
    plt.xlabel('Year', fontsize=15, fontweight='normal')  # 变大
    plt.ylabel('Global Ocean SO₂ Emission', fontsize=15)
    
    # 坐标轴刻度字体
    plt.xticks(fontsize=13)
    plt.yticks(fontsize=13)
    
    plt.tight_layout()
    plt.savefig(out_path, bbox_inches='tight')
    plt.close()
    print(f'Saved time series: {out_path}')
# ====================================================================================


def main() -> None:
    os.makedirs(os.path.join(OUT_DIR, 'figs'), exist_ok=True)
    os.makedirs(os.path.join(OUT_DIR, 'processed_data'), exist_ok=True)

    years = np.arange(YEAR_START, YEAR_END + 1, dtype=np.int16)
    months = np.arange(1, 13, dtype=np.int16)
    ocean_mask = _get_ocean_mask_1deg()
    all_year_month_grids = []

    # 存储每年全球平均
    annual_global_means = []

    for year in years.astype(int):
        print(f'Loading monthly SOx for {year}...')
        monthly_grids = load_monthly_1deg(year)
        monthly_grids = np.where(ocean_mask[None, :, :], monthly_grids, np.nan)
        monthly_grids = np.stack(
            [_fill_missing_ocean_sox_in_trop_midlat(grid, ocean_mask) for grid in monthly_grids],
            axis=0,
        )
        all_year_month_grids.append(monthly_grids.astype(np.float32))
        print(f'{year}: finite cells per month = {[int(np.count_nonzero(np.isfinite(g))) for g in monthly_grids]}')

        # 计算年均 → 全球平均
        annual_mean = np.nanmean(monthly_grids, axis=0)
        global_mean = np.nanmean(annual_mean)
        annual_global_means.append(global_mean)

    norm = _build_shared_norm(all_year_month_grids)
    for year, monthly_grids in zip(years.astype(int), all_year_month_grids):
        plot_year_mean(year, monthly_grids, norm)

    # 画年际变化曲线
    plot_global_annual_mean_timeseries(years, annual_global_means)

    # ====================== 【开关控制是否保存 npz】 ======================
    if SAVE_NPZ:
        monthly_sox_grids = np.stack(all_year_month_grids, axis=0)
        out_npz = os.path.join(
            OUT_DIR,
            'processed_data',
            f'sox_monthly_{YEAR_START}_{YEAR_END}.npz',
        )
        np.savez_compressed(
            out_npz,
            years=years,
            months=months,
            lat_edges=LAT_EDGES.astype(np.float32),
            lon_edges=LON_EDGES.astype(np.float32),
            ocean_mask=ocean_mask.astype(np.bool_),
            monthly_sox_grids=monthly_sox_grids.astype(np.float32),
        )
        print(f'Saved data: {out_npz}')
        print(f'monthly_sox_grids shape: {monthly_sox_grids.shape}')
    else:
        print("⏩ SAVE_NPZ = False，跳过保存 .npz")
    # ====================================================================


if __name__ == '__main__':
    main()