import os
import numpy as np
import pandas as pd
import netCDF4 as nc
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib import gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature


CSV_PATH = '/home/chenyiqi/260320_ship_emission/ship_emission_LH/spatial_2019_2020_month.csv'
OUT_DIR = '/home/chenyiqi/260320_ship_emission'
YEAR_A = 2020
YEAR_B = 2019
LSMASK_PATH = '/data/chenyiqi/251007_tropic/landsea.nc'

# Set to 'log' for logarithmic color scaling, or 'linear' for linear color scaling.
COLOR_SCALE_MODE = 'log'

# Shared color settings for all three subplots.
SHARED_CMAP = 'viridis'
SHARED_VMIN = 10**-3.5
SHARED_VMAX = 10**2.5
LAT_EDGES = np.arange(-90, 91, 1)
LON_EDGES = np.arange(-180, 181, 1)
OUT_DIFF_NPZ = os.path.join(
	OUT_DIR,
	f'processed_data/soxdiff_monthly_{YEAR_B}m{YEAR_A}.npz'
)
OCEAN_MASK_1DEG = None


def _build_output_paths(year_a, year_b, month):
	compare_path = os.path.join(
		OUT_DIR,
		f'figs/sox_{year_b}_{month:02d}_minus_{year_a}_{month:02d}_shared_scale.png'
	)
	return compare_path


def load_emission_csv(csv_path):
	expected_cols = ['year', 'month', 'lon', 'lat', 'PM', 'NOx', 'SOx', 'CO', 'HC', 'CO2', 'N2O', 'CH4', 'BC']

	# Try normal header parsing first, then fallback for malformed/headerless files.
	df = pd.read_csv(csv_path)
	if not {'year', 'month', 'lon', 'lat', 'SOx'}.issubset(df.columns):
		df = pd.read_csv(
			csv_path,
			header=None,
			names=expected_cols,
			on_bad_lines='skip'
		)

	for col in ['year', 'month', 'lon', 'lat', 'SOx']:
		df[col] = pd.to_numeric(df[col], errors='coerce')

	df = df.dropna(subset=['year', 'month', 'lon', 'lat', 'SOx'])
	return df


def _aggregate_to_1deg(df, year, month):
	data = df[(df['year'] == year) & (df['month'] == month)].copy()
	if data.empty:
		raise ValueError(f'No records found for year={year}, month={month}.')

	data['lon_bin'] = np.floor(data['lon']).astype(int)
	data['lat_bin'] = np.floor(data['lat']).astype(int)
	data['lon_bin'] = data['lon_bin'].clip(-180, 179)
	data['lat_bin'] = data['lat_bin'].clip(-90, 89)

	grid_sum = (
		data.groupby(['lat_bin', 'lon_bin'], as_index=False)['SOx']
		.sum()
	)
	return data, grid_sum


def _add_gridlines(ax):
	gl = ax.gridlines(draw_labels=True, linewidth=0.4, linestyle='--', alpha=0.35)
	# Keep labels only on left and bottom to avoid duplicated labels on top/right.
	gl.top_labels = False
	gl.right_labels = False
	return gl


def _build_ocean_mask_1deg():
	"""Build 1x1 degree ocean mask from LSMASK file, excluding |lat| > 60."""
	with nc.Dataset(LSMASK_PATH, 'r') as ds:
		lat_lsm = ds.variables['lat'][:].astype(float)
		lon_lsm = ds.variables['lon'][:].astype(float)
		lsmask = ds.variables['LSMASK'][:].astype(bool)

	# Match load_land_sea_mask processing in mod08_to_npz.py.
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
	polar_rows = (lat_grid[:, 0] > 60.0) | (lat_grid[:, 0] < -60.0)
	ocean_native[polar_rows, :] = False

	lat_native = lat_grid[:, 0]
	lon_native = lon_grid[0, :]
	lat_centers = (LAT_EDGES[:-1] + LAT_EDGES[1:]) / 2.0
	lon_centers = (LON_EDGES[:-1] + LON_EDGES[1:]) / 2.0

	lat_idx = np.abs(lat_native[:, None] - lat_centers[None, :]).argmin(axis=0)
	lon_dist = np.abs(((lon_native[:, None] - lon_centers[None, :]) + 180.0) % 360.0 - 180.0)
	lon_idx = lon_dist.argmin(axis=0)

	ocean_mask_1deg = ocean_native[np.ix_(lat_idx, lon_idx)]
	ocean_mask_1deg[(lat_centers < -60.0) | (lat_centers > 60.0), :] = False
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


def compute_sox_difference_grid(df, year_a, year_b, month):
	data_a, grid_sum_a = _aggregate_to_1deg(df, year=year_a, month=month)
	data_b, grid_sum_b = _aggregate_to_1deg(df, year=year_b, month=month)

	lat_edges = LAT_EDGES
	lon_edges = LON_EDGES
	sox_grid_a = np.full((len(lat_edges) - 1, len(lon_edges) - 1), np.nan, dtype=float)
	sox_grid_b = np.full((len(lat_edges) - 1, len(lon_edges) - 1), np.nan, dtype=float)

	lat_idx_a = (grid_sum_a['lat_bin'] - (-90)).to_numpy()
	lon_idx_a = (grid_sum_a['lon_bin'] - (-180)).to_numpy()
	sox_grid_a[lat_idx_a, lon_idx_a] = grid_sum_a['SOx'].to_numpy()

	lat_idx_b = (grid_sum_b['lat_bin'] - (-90)).to_numpy()
	lon_idx_b = (grid_sum_b['lon_bin'] - (-180)).to_numpy()
	sox_grid_b[lat_idx_b, lon_idx_b] = grid_sum_b['SOx'].to_numpy()

	ocean_mask = _get_ocean_mask_1deg()
	sox_grid_a = np.where(ocean_mask, sox_grid_a, np.nan)
	sox_grid_b = np.where(ocean_mask, sox_grid_b, np.nan)
	sox_grid_a = _fill_missing_ocean_sox_in_trop_midlat(sox_grid_a, ocean_mask)
	sox_grid_b = _fill_missing_ocean_sox_in_trop_midlat(sox_grid_b, ocean_mask)

	eligible_mask = np.isfinite(sox_grid_a) & np.isfinite(sox_grid_b)
	diff_grid = np.full_like(sox_grid_a, np.nan, dtype=float)
	diff_grid[eligible_mask] = sox_grid_b[eligible_mask] - sox_grid_a[eligible_mask]

	if not np.any(np.isfinite(diff_grid)):
		raise ValueError(f'No grid cells with finite values in both years for month={month}.')

	return {
		'data_a_count': len(data_a),
		'data_b_count': len(data_b),
		'sox_grid_a': sox_grid_a,
		'sox_grid_b': sox_grid_b,
		'diff_grid': diff_grid,
		'eligible_mask': eligible_mask,
		'ocean_mask': ocean_mask,
	}


def plot_sox_difference(df, year_a, year_b, month, sox_grid_a, sox_grid_b, diff_grid, eligible_mask):
	color_mode = COLOR_SCALE_MODE.lower().strip()
	if color_mode not in {'log', 'linear'}:
		raise ValueError("COLOR_SCALE_MODE must be either 'log' or 'linear'.")

	out_fig_compare = _build_output_paths(year_a, year_b, month)

	lat_edges = LAT_EDGES
	lon_edges = LON_EDGES

	os.makedirs(os.path.join(OUT_DIR, 'figs'), exist_ok=True)
	os.makedirs(os.path.join(OUT_DIR, 'processed_data'), exist_ok=True)

	finite_diff = diff_grid[np.isfinite(diff_grid)]
	if finite_diff.size == 0:
		raise ValueError('No grid cells with finite values in both years.')

	if color_mode == 'log':
		shared_norm = LogNorm(vmin=SHARED_VMIN, vmax=SHARED_VMAX)
	else:
		shared_norm = Normalize(vmin=SHARED_VMIN, vmax=SHARED_VMAX)

	# Three maps share one fixed color scale and one colorbar.
	sox_grid_a_ocean = sox_grid_a
	sox_grid_b_ocean = sox_grid_b
	diff_plot = diff_grid.copy()

	if color_mode == 'log':
		# LogNorm requires strictly positive values.
		diff_plot = np.where(diff_plot > 0, diff_plot, np.nan)
		plot_a = np.where(sox_grid_a_ocean > 0, sox_grid_a_ocean, np.nan)
		plot_b = np.where(sox_grid_b_ocean > 0, sox_grid_b_ocean, np.nan)
	else:
		plot_a = sox_grid_a_ocean
		plot_b = sox_grid_b_ocean

	fig3 = plt.figure(figsize=(12, 14), dpi=300)
	gs = gridspec.GridSpec(
		nrows=3,
		ncols=2,
		width_ratios=[0.97, 0.03],
		hspace=0.14,
		wspace=0.0,
	)
	axes = [
		fig3.add_subplot(gs[0, 0], projection=ccrs.PlateCarree()),
		fig3.add_subplot(gs[1, 0], projection=ccrs.PlateCarree()),
		fig3.add_subplot(gs[2, 0], projection=ccrs.PlateCarree()),
	]
	cax_shared = fig3.add_subplot(gs[:, 1])

	hb_diff = axes[0].pcolormesh(
		lon_edges,
		lat_edges,
		diff_plot,
		cmap=SHARED_CMAP,
		norm=shared_norm,
		shading='auto',
		transform=ccrs.PlateCarree()
	)
	axes[0].add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none', zorder=2)
	axes[0].coastlines(resolution='110m', linewidth=0.6, color='black', zorder=3)
	_add_gridlines(axes[0])
	axes[0].set_xlim(-180, 180)
	axes[0].set_ylim(-90, 90)
	axes[0].set_title(f'SOx Difference ({year_b}-{month:02d} minus {year_a}-{month:02d})')

	axes[1].pcolormesh(
		lon_edges,
		lat_edges,
		plot_b,
		cmap=SHARED_CMAP,
		norm=shared_norm,
		shading='auto',
		transform=ccrs.PlateCarree()
	)
	axes[1].add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none', zorder=2)
	axes[1].coastlines(resolution='110m', linewidth=0.6, color='black', zorder=3)
	_add_gridlines(axes[1])
	axes[1].set_xlim(-180, 180)
	axes[1].set_ylim(-90, 90)
	axes[1].set_title(f'SOx {year_b}-{month:02d}')

	axes[2].pcolormesh(
		lon_edges,
		lat_edges,
		plot_a,
		cmap=SHARED_CMAP,
		norm=shared_norm,
		shading='auto',
		transform=ccrs.PlateCarree()
	)
	axes[2].add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none', zorder=2)
	axes[2].coastlines(resolution='110m', linewidth=0.6, color='black', zorder=3)
	_add_gridlines(axes[2])
	axes[2].set_xlim(-180, 180)
	axes[2].set_ylim(-90, 90)
	axes[2].set_title(f'SOx {year_a}-{month:02d}')

	for axx in axes:
		axx.set_xlabel('Longitude')
		axx.set_ylabel('Latitude')

	cbar_shared = fig3.colorbar(
		plt.cm.ScalarMappable(norm=shared_norm, cmap=SHARED_CMAP),
		cax=cax_shared,
		extend='both',
		pad=0.0,
	)
	cbar_shared.set_label('ton per 1°*1° grid')
	cax_shared.tick_params(labelsize=8)

	fig3.savefig(out_fig_compare, bbox_inches='tight')
	plt.close(fig3)

	print(f'Saved figure: {out_fig_compare}')
	print(f'Grid cells with finite difference: {np.count_nonzero(eligible_mask):,}')


if __name__ == '__main__':
	df_all = load_emission_csv(CSV_PATH)
	os.makedirs(os.path.join(OUT_DIR, 'figs'), exist_ok=True)
	os.makedirs(os.path.join(OUT_DIR, 'processed_data'), exist_ok=True)
	months = np.arange(1, 13, dtype=np.int16)
	month_diff_grids = []

	for month in range(1, 13):
		month_data = compute_sox_difference_grid(
			df_all,
			year_a=YEAR_A,
			year_b=YEAR_B,
			month=month,
		)
		plot_sox_difference(
			df_all,
			year_a=YEAR_A,
			year_b=YEAR_B,
			month=month,
			sox_grid_a=month_data['sox_grid_a'],
			sox_grid_b=month_data['sox_grid_b'],
			diff_grid=month_data['diff_grid'],
			eligible_mask=month_data['eligible_mask'],
		)
		month_diff_grids.append(month_data['diff_grid'].astype(np.float32))
		print(f'Records used for {YEAR_A}-{month:02d}: {month_data["data_a_count"]:,}')
		print(f'Records used for {YEAR_B}-{month:02d}: {month_data["data_b_count"]:,}')

	np.savez_compressed(
		OUT_DIFF_NPZ,
		months=months,
		year_a=np.int16(YEAR_A),
		year_b=np.int16(YEAR_B),
		lat_edges=LAT_EDGES.astype(np.float32),
		lon_edges=LON_EDGES.astype(np.float32),
		ocean_mask=_get_ocean_mask_1deg(),
		diff_grids=np.stack(month_diff_grids, axis=0),
	)
	print(f'Saved data: {OUT_DIFF_NPZ}')
