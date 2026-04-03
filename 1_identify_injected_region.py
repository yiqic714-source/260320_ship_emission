import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm, Normalize
from matplotlib import gridspec
import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.prepared import prep
from shapely.ops import unary_union


CSV_PATH = '/home/chenyiqi/260320_ship_emission/ship_emission_LH/spatial_2019_2020_month.csv'
OUT_DIR = '/home/chenyiqi/260320_ship_emission'
YEAR_A = 2020
YEAR_B = 2019
# Percentage (0-100] of largest differences to highlight.
TOP_PERCENT = 10
OUT_POINTS_CSV = os.path.join(
	OUT_DIR,
	f'processed_data/sox_selected_points_{YEAR_B}_minus_{YEAR_A}_top{TOP_PERCENT:g}.csv'
)

# Set to 'log' for logarithmic color scaling, or 'linear' for linear color scaling.
COLOR_SCALE_MODE = 'log'

# Maximum number of highlighted top-10% grid cells to draw as dots.
MAX_TOP10_DOTS = 600

# Shared color settings for all three subplots.
SHARED_CMAP = 'viridis'
SHARED_VMIN = 10**-3.5
SHARED_VMAX = 10**2.5
LAT_EDGES = np.arange(-90, 91, 1)
LON_EDGES = np.arange(-180, 181, 1)


def _build_output_paths(year_a, year_b, month):
	compare_path = os.path.join(
		OUT_DIR,
		f'figs/sox_{year_b}_{month:02d}_minus_{year_a}_{month:02d}_shared_scale.png'
	)
	joint_pdf_path = os.path.join(
		OUT_DIR,
		f'figs/sox_joint_pdf_{year_a}_{month:02d}_{year_b}_{month:02d}_top{TOP_PERCENT:g}.png'
	)
	return compare_path, joint_pdf_path


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


def _build_land_mask(lat_edges, lon_edges):
	land_path = shpreader.natural_earth(
		resolution='110m',
		category='physical',
		name='land'
	)
	land_reader = shpreader.Reader(land_path)
	land_geom = unary_union(list(land_reader.geometries()))
	land_prepared = prep(land_geom)

	lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
	lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
	land_mask = np.zeros((len(lat_centers), len(lon_centers)), dtype=bool)

	for i, lat in enumerate(lat_centers):
		for j, lon in enumerate(lon_centers):
			land_mask[i, j] = land_prepared.covers(Point(float(lon), float(lat)))

	return land_mask


def _build_coastal_exclusion_mask(land_mask, distance_cells=2):
	# Exclude land and ocean cells within <= distance_cells grid steps from land.
	if distance_cells < 0:
		raise ValueError('distance_cells must be >= 0.')

	excluded = land_mask.copy()
	if distance_cells == 0:
		return excluded

	padded = np.pad(land_mask, distance_cells, mode='constant', constant_values=False)
	for di in range(-distance_cells, distance_cells + 1):
		for dj in range(-distance_cells, distance_cells + 1):
			window = padded[
				distance_cells + di:distance_cells + di + land_mask.shape[0],
				distance_cells + dj:distance_cells + dj + land_mask.shape[1],
			]
			excluded |= window

	return excluded


def _sparsify_points(lons, lats, max_points=1200):
	if len(lons) <= max_points:
		return lons, lats
	idx = np.linspace(0, len(lons) - 1, max_points, dtype=int)
	return lons[idx], lats[idx]


def _add_gridlines(ax):
	gl = ax.gridlines(draw_labels=True, linewidth=0.4, linestyle='--', alpha=0.35)
	# Keep labels only on left and bottom to avoid duplicated labels on top/right.
	gl.top_labels = False
	gl.right_labels = False
	return gl


def _calc_r2(x, y):
	x = np.asarray(x, dtype=float)
	y = np.asarray(y, dtype=float)
	if x.size < 2 or y.size < 2:
		return np.nan
	if np.allclose(np.std(x), 0) or np.allclose(np.std(y), 0):
		return np.nan
	r = np.corrcoef(x, y)[0, 1]
	return float(r * r)


def _plot_joint_pdf_combined(all_a, all_b, top_a, top_b, out_path, month, year_a, year_b):
	all_a = np.asarray(all_a, dtype=float)
	all_b = np.asarray(all_b, dtype=float)
	top_a = np.asarray(top_a, dtype=float)
	top_b = np.asarray(top_b, dtype=float)

	all_mask = np.isfinite(all_a) & np.isfinite(all_b) & (all_a > 0) & (all_b > 0)
	top_mask = np.isfinite(top_a) & np.isfinite(top_b) & (top_a > 0) & (top_b > 0)

	all_x = np.log10(all_a[all_mask])
	all_y = np.log10(all_b[all_mask])
	top_x = np.log10(top_a[top_mask])
	top_y = np.log10(top_b[top_mask])

	if all_x.size == 0:
		raise ValueError('No positive finite values available for all-data joint distribution.')
	if top_x.size == 0:
		raise ValueError('No positive finite values available for top-percent joint distribution.')

	xbins_top = np.linspace(top_x.min(), top_x.max(), 55)
	ybins_top = np.linspace(top_y.min(), top_y.max(), 55)
	xbins_all = np.linspace(all_x.min(), all_x.max(), 55)
	ybins_all = np.linspace(all_y.min(), all_y.max(), 55)

	h_top, _, _ = np.histogram2d(top_x, top_y, bins=[xbins_top, ybins_top], density=True)
	h_all, _, _ = np.histogram2d(all_x, all_y, bins=[xbins_all, ybins_all], density=True)

	all_positive = h_all[h_all > 0]
	top_positive = h_top[h_top > 0]
	if all_positive.size == 0 or top_positive.size == 0:
		raise ValueError('Joint distribution bins are empty after filtering.')

	all_joint_positive = h_all[h_all > 0]
	top_joint_positive = h_top[h_top > 0]
	joint_vmin = min(all_joint_positive.min(), top_joint_positive.min())
	joint_vmax = max(all_joint_positive.max(), top_joint_positive.max())
	joint_norm = LogNorm(vmin=joint_vmin, vmax=joint_vmax)

	fig, axes = plt.subplots(2, 2, figsize=(12, 9), dpi=300, sharex='col')
	ax_joint_top, ax_joint_all = axes[0, 0], axes[0, 1]
	ax_pdf_top, ax_pdf_all = axes[1, 0], axes[1, 1]

	mappable_top = ax_joint_top.pcolormesh(
		xbins_top,
		ybins_top,
		h_top.T,
		cmap='Blues',
		norm=joint_norm,
		shading='auto',
	)
	ax_joint_all.pcolormesh(
		xbins_all,
		ybins_all,
		h_all.T,
		cmap='Blues',
		norm=joint_norm,
		shading='auto',
	)

	r2_top = _calc_r2(top_x, top_y)
	r2_all = _calc_r2(all_x, all_y)
	for ax, title, r2 in [
		(ax_joint_top, f'Joint Distribution Top {TOP_PERCENT:g}%', r2_top),
		(ax_joint_all, 'Joint Distribution All Data', r2_all),
	]:
		ax.set_xlabel(f'SOx {year_a}-{month:02d}')
		ax.set_ylabel(f'SOx {year_b}-{month:02d}')
		ax.set_title(title)
		ax.grid(True, which='both', linestyle='--', alpha=0.3)
		r2_text = f'R2={r2:.3f}' if np.isfinite(r2) else 'R2=NA'
		ax.text(
			0.02,
			0.98,
			r2_text,
			transform=ax.transAxes,
			va='top',
			ha='left',
			fontsize=9,
			bbox=dict(facecolor='white', edgecolor='none', alpha=0.7),
		)

	def _plot_pdf_pair(ax, vals_a, vals_b, title):
		log_min = min(vals_a.min(), vals_b.min())
		log_max = max(vals_a.max(), vals_b.max())
		pdf_bins = np.linspace(log_min, log_max, 70)
		pdf_centers = 0.5 * (pdf_bins[:-1] + pdf_bins[1:])
		hist_a, _ = np.histogram(vals_a, bins=pdf_bins, density=True)
		hist_b, _ = np.histogram(vals_b, bins=pdf_bins, density=True)
		valid_a = hist_a > 0
		valid_b = hist_b > 0
		ax.plot(
			pdf_centers[valid_a],
			hist_a[valid_a],
			color='#d95f02',
			lw=1.8,
			label=f'{year_a}-{month:02d} (n={vals_a.size:,})',
		)
		ax.plot(
			pdf_centers[valid_b],
			hist_b[valid_b],
			color='#1b9e77',
			lw=1.8,
			label=f'{year_b}-{month:02d} (n={vals_b.size:,})',
		)
		ax.set_title(title)
		ax.set_xlabel('log10(SOx) [ton per 1°*1° grid]')
		ax.set_ylabel('PDF')
		ax.grid(True, which='both', linestyle='--', alpha=0.35)
		ax.legend(frameon=False)

	_plot_pdf_pair(ax_pdf_top, top_x, top_y, f'PDF Top {TOP_PERCENT:g}%')
	_plot_pdf_pair(ax_pdf_all, all_x, all_y, 'PDF All Data')

	fig.tight_layout(rect=[0, 0, 1, 0.98])
	fig.savefig(out_path, bbox_inches='tight')
	plt.close(fig)


def plot_sox_difference(df, year_a, year_b, month, land_mask, coastal_exclusion_mask):
	color_mode = COLOR_SCALE_MODE.lower().strip()
	if color_mode not in {'log', 'linear'}:
		raise ValueError("COLOR_SCALE_MODE must be either 'log' or 'linear'.")
	if not (0 < TOP_PERCENT <= 100):
		raise ValueError('TOP_PERCENT must be in the range (0, 100].')

	data_a, grid_sum_a = _aggregate_to_1deg(df, year=year_a, month=month)
	data_b, grid_sum_b = _aggregate_to_1deg(df, year=year_b, month=month)
	out_fig_compare, out_fig_joint_pdf = _build_output_paths(year_a, year_b, month)

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

	# Enforce "filter first, then top-10%": only ocean/offshore cells with valid values in both years.
	eligible_mask = (
		~coastal_exclusion_mask
		& np.isfinite(sox_grid_a)
		& np.isfinite(sox_grid_b)
	)
	diff_grid = np.full_like(sox_grid_a, np.nan, dtype=float)
	diff_grid[eligible_mask] = sox_grid_b[eligible_mask] - sox_grid_a[eligible_mask]

	os.makedirs(OUT_DIR, exist_ok=True)

	finite_diff = diff_grid[np.isfinite(diff_grid)]
	if finite_diff.size == 0:
		raise ValueError('No ocean grid cells available after land masking.')

	if color_mode == 'log':
		shared_norm = LogNorm(vmin=SHARED_VMIN, vmax=SHARED_VMAX)
	else:
		shared_norm = Normalize(vmin=SHARED_VMIN, vmax=SHARED_VMAX)


	ocean_valid = eligible_mask
	eligible_diff = diff_grid[eligible_mask]
	n_eligible = eligible_diff.size
	if n_eligible == 0:
		raise ValueError('No eligible ocean cells available for top-percent selection.')
	top_count = max(1, int(np.floor(n_eligible * TOP_PERCENT / 100.0)))
	eligible_i, eligible_j = np.where(eligible_mask)
	order = np.argsort(eligible_diff)
	top_idx = order[-top_count:]
	top10_mask = np.zeros_like(eligible_mask, dtype=bool)
	top10_mask[eligible_i[top_idx], eligible_j[top_idx]] = True
	percentile_threshold = float(np.min(eligible_diff[top_idx]))
	lon_centers = (lon_edges[:-1] + lon_edges[1:]) / 2
	lat_centers = (lat_edges[:-1] + lat_edges[1:]) / 2
	top10_i, top10_j = np.where(top10_mask)
	top10_lons = lon_centers[top10_j]
	top10_lats = lat_centers[top10_i]
	top10_lons_plot, top10_lats_plot = _sparsify_points(
		top10_lons,
		top10_lats,
		max_points=MAX_TOP10_DOTS,
	)

	# Three maps share one fixed color scale and one colorbar.
	sox_grid_a_ocean = sox_grid_a.copy()
	sox_grid_b_ocean = sox_grid_b.copy()
	sox_grid_a_ocean[coastal_exclusion_mask] = np.nan
	sox_grid_b_ocean[coastal_exclusion_mask] = np.nan
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

	hb_year_b = axes[1].pcolormesh(
		lon_edges,
		lat_edges,
		plot_b,
		cmap=SHARED_CMAP,
		norm=shared_norm,
		shading='auto',
		transform=ccrs.PlateCarree()
	)
	axes[1].scatter(
		top10_lons_plot,
		top10_lats_plot,
		s=1.2,
		c='red',
		alpha=0.55,
		linewidths=0,
		marker='o',
		transform=ccrs.PlateCarree(),
		zorder=4,
		label=f'Top {TOP_PERCENT:g}% difference (sparse dots)'
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
	axes[2].scatter(
		top10_lons_plot,
		top10_lats_plot,
		s=1.2,
		c='red',
		alpha=0.55,
		linewidths=0,
		marker='o',
		transform=ccrs.PlateCarree(),
		zorder=4,
		label=f'Top {TOP_PERCENT:g}% difference (sparse dots)'
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
	print(f'Records used for {year_a}-{month:02d}: {len(data_a):,}')
	print(f'Records used for {year_b}-{month:02d}: {len(data_b):,}')
	print(f'Number of ocean cells with finite difference: {np.count_nonzero(ocean_valid):,}')
	print(f'Top {TOP_PERCENT:g}% threshold value: {percentile_threshold:.6g}')
	print(
		f'Top {TOP_PERCENT:g}% cells total: {len(top10_lons):,} '
		f'({len(top10_lons)/n_eligible:.3%} of eligible); plotted: {len(top10_lons_plot):,}'
	)
	total_diff_sum = float(np.nansum(eligible_diff))
	top_diff_sum = float(np.nansum(eligible_diff[top_idx]))
	if not np.isclose(total_diff_sum, 0.0):
		top_share_pct = 100.0 * top_diff_sum / total_diff_sum
		print(
			f'Top {TOP_PERCENT:g}% diff sum: {top_diff_sum:.6g}; '
			f'all diff sum: {total_diff_sum:.6g}; '
			f'share: {top_share_pct:.2f}%'
		)
	else:
		print(
			f'Top {TOP_PERCENT:g}% diff sum: {top_diff_sum:.6g}; '
			f'all diff sum is ~0, share is undefined.'
		)

	_plot_joint_pdf_combined(
		all_a=sox_grid_a_ocean[ocean_valid],
		all_b=sox_grid_b_ocean[ocean_valid],
		top_a=sox_grid_a_ocean[top10_mask],
		top_b=sox_grid_b_ocean[top10_mask],
		out_path=out_fig_joint_pdf,
		month=month,
		year_a=year_a,
		year_b=year_b,
	)
	print(f'Saved figure: {out_fig_joint_pdf}')

	selected_points = pd.DataFrame(
		{
			'month': np.full(len(top10_lons), month, dtype=int),
			'lon': top10_lons,
			'lat': top10_lats,
			f'sox_{year_a}': sox_grid_a[top10_mask],
			f'sox_{year_b}': sox_grid_b[top10_mask],
			f'diff_{year_b}_minus_{year_a}': diff_grid[top10_mask],
		}
	)
	return selected_points


if __name__ == '__main__':
	df_all = load_emission_csv(CSV_PATH)
	os.makedirs(OUT_DIR, exist_ok=True)
	land_mask = _build_land_mask(LAT_EDGES, LON_EDGES)
	coastal_exclusion_mask = _build_coastal_exclusion_mask(land_mask, distance_cells=2)
	selected_points_all = []

	for month in range(1, 13):
		selected_points = plot_sox_difference(
			df_all,
			year_a=YEAR_A,
			year_b=YEAR_B,
			month=month,
			land_mask=land_mask,
			coastal_exclusion_mask=coastal_exclusion_mask,
		)
		selected_points_all.append(selected_points)

	selected_points_df = pd.concat(selected_points_all, ignore_index=True)
	selected_points_df.to_csv(OUT_POINTS_CSV, index=False)
	print(f'Saved selected points: {OUT_POINTS_CSV}')
