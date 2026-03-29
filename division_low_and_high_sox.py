import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
from matplotlib.colors import ListedColormap, BoundaryNorm
from matplotlib.patches import Patch
import cartopy.crs as ccrs
import cartopy.feature as cfeature


CSV_PATH = '/home/chenyiqi/260320_ship_emission/ship_emission_LH/spatial_2019_2020_month.csv'
OUT_DIR = '/home/chenyiqi/260320_ship_emission/figs'
OUT_FIG = os.path.join(OUT_DIR, 'sox_spatial_2020_03.png')
OUT_FIG_THRESHOLD = os.path.join(OUT_DIR, 'sox_spatial_2020_03_threshold.png')


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


def plot_sox_spatial(df, year=2020, month=3):
	data = df[(df['year'] == year) & (df['month'] == month)].copy()
	if data.empty:
		raise ValueError(f'No records found for year={year}, month={month}.')

	# Aggregate to 1 degree x 1 degree cells.
	data['lon_bin'] = np.floor(data['lon']).astype(int)
	data['lat_bin'] = np.floor(data['lat']).astype(int)
	data['lon_bin'] = data['lon_bin'].clip(-180, 179)
	data['lat_bin'] = data['lat_bin'].clip(-90, 89)

	grid_sum = (
		data.groupby(['lat_bin', 'lon_bin'], as_index=False)['SOx']
		.sum()
	)

	lat_edges = np.arange(-90, 91, 1)
	lon_edges = np.arange(-180, 181, 1)
	sox_grid = np.full((len(lat_edges) - 1, len(lon_edges) - 1), np.nan)

	lat_idx = (grid_sum['lat_bin'] - (-90)).to_numpy()
	lon_idx = (grid_sum['lon_bin'] - (-180)).to_numpy()
	sox_grid[lat_idx, lon_idx] = grid_sum['SOx'].to_numpy()

	os.makedirs(OUT_DIR, exist_ok=True)

	fig, ax = plt.subplots(
		figsize=(12, 5),
		dpi=300,
		subplot_kw={'projection': ccrs.PlateCarree()}
	)

	positive = np.isfinite(sox_grid) & (sox_grid > 0)
	if positive.any():
		vmin = np.nanmin(sox_grid[positive])
		vmax = np.nanmax(sox_grid[positive])
		norm = LogNorm(vmin=max(vmin, 1e-12), vmax=vmax)
	else:
		norm = None

	hb = ax.pcolormesh(
		lon_edges,
		lat_edges,
		sox_grid,
		cmap='YlOrRd',
		norm=norm,
		shading='auto',
		transform=ccrs.PlateCarree()
	)

	ax.add_feature(cfeature.LAND, facecolor='none', edgecolor='none', zorder=2)
	ax.coastlines(resolution='110m', linewidth=0.6, color='black', zorder=3)
	ax.gridlines(draw_labels=True, linewidth=0.4, linestyle='--', alpha=0.35)

	ax.set_xlim(-180, 180)
	ax.set_ylim(-90, 90)
	ax.set_xlabel('Longitude')
	ax.set_ylabel('Latitude')
	ax.set_title('SOx Spatial Distribution (2020-03, 1°x1°)')

	cbar = fig.colorbar(hb, ax=ax, pad=0.02)
	cbar.set_label('SOx emission (ton per 1°x1° grid cell)')

	fig.tight_layout()
	fig.savefig(OUT_FIG, bbox_inches='tight')
	plt.close(fig)

	# Second map: threshold classes only
	threshold_grid = np.full_like(sox_grid, np.nan, dtype=float)
	threshold_grid[sox_grid > 5] = 1.0
	threshold_grid[sox_grid < 1] = -1.0

	fig2, ax2 = plt.subplots(
		figsize=(12, 5),
		dpi=300,
		subplot_kw={'projection': ccrs.PlateCarree()}
	)

	thr_cmap = ListedColormap(['royalblue', 'red'])
	thr_norm = BoundaryNorm([-1.5, 0.0, 1.5], thr_cmap.N)

	ax2.pcolormesh(
		lon_edges,
		lat_edges,
		threshold_grid,
		cmap=thr_cmap,
		norm=thr_norm,
		shading='auto',
		transform=ccrs.PlateCarree()
	)

	ax2.coastlines(resolution='110m', linewidth=0.6, color='black', zorder=3)
	ax2.gridlines(draw_labels=True, linewidth=0.4, linestyle='--', alpha=0.35)
	ax2.set_xlim(-180, 180)
	ax2.set_ylim(-90, 90)
	ax2.set_xlabel('Longitude')
	ax2.set_ylabel('Latitude')
	ax2.set_title('SOx Threshold Map (2020-03, 1°x1°)')

	legend_items = [
		Patch(facecolor='red', edgecolor='red', label='high SOx'),
		Patch(facecolor='royalblue', edgecolor='royalblue', label='low SOx'),
	]
	ax2.legend(handles=legend_items, loc='lower left', frameon=True)

	fig2.tight_layout()
	fig2.savefig(OUT_FIG_THRESHOLD, bbox_inches='tight')
	plt.close(fig2)

	print(f'Saved figure: {OUT_FIG}')
	print(f'Saved figure: {OUT_FIG_THRESHOLD}')
	print(f'Number of records used: {len(data):,}')
	print(f'Number of 1°x1° cells with emissions: {len(grid_sum):,}')


if __name__ == '__main__':
	df_all = load_emission_csv(CSV_PATH)
	plot_sox_spatial(df_all, year=2020, month=3)
