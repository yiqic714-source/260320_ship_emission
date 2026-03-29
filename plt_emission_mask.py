import argparse
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
from cartopy.io import shapereader
import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from scipy.spatial import cKDTree
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep


# Shared piecewise lon-lat windows (lon_min, lon_max, lat_min, lat_max).
REGIONAL_WINDOW_SEGMENTS = [
	(130.0, 135.0, -60.0, 15.0),
	(135.0, 140.0, -60.0, 25.0),
	(140.0, 163.0, -60.0, 32.0),
	(-96.0, -81.0, -60.0, 15.0),
	(163.0, 180.0, -60.0, 60.0),
	(-180.0, -96.0, -60.0, 60.0),
	(-81.0, -60.0, -60.0, 10.0),
	(-60.0, -20.0, -60.0, 28.0),
	(-20.0, 130.0, -60.0, 2.0),
]


def parse_args() -> argparse.Namespace:
	year = 2021
	parser = argparse.ArgumentParser(
		description="Plot global PM distribution over ocean on a 1-degree grid."
	)
	parser.add_argument(
		"--value-column",
		default="PM",
		help="CSV column name to process (default: PM)",
	)
	parser.add_argument(
		"csv_file",
		nargs="?",
		default=f"ship_emission_LH/L01dA_{year}_global.csv",
		help=f"Input CSV file path (default: L01dA_{year}_global.csv)",
	)
	parser.add_argument(
		"--output",
		default=f"figs/pm_global_distribution_{year}.png",
		help=f"Output image path (default: figs/pm_global_distribution_{year}.png)",
	)
	parser.add_argument(
		"--mask-output",
		default=f"ocean_mask_{year}.npz",
		help=f"Output NPZ path for reusable ocean mask (default: ocean_mask_{year}.npz)",
	)
	parser.add_argument(
		"--mask-plot-output",
		default=f"figs/ocean_mask_{year}.png",
		help=f"Output image path for reusable ocean mask plot (default: figs/ocean_mask_{year}.png)",
	)
	parser.add_argument(
		"--pdf-output",
		default=None,
		help="Output image path for the pre-screening probability density plot",
	)
	parser.add_argument(
		"--shading",
		default="auto",
		choices=["auto", "nearest", "flat", "gouraud"],
		help="Shading mode for pcolormesh (default: auto)",
	)
	return parser.parse_args()


def percentile_limits(values: np.ndarray, low: float = 1, high: float = 99) -> tuple[float, float]:
	finite_values = values[np.isfinite(values)]
	if finite_values.size == 0:
		return 0.0, 1.0
	vmin, vmax = np.percentile(finite_values, [low, high])
	if vmin == vmax:
		vmax = vmin + 1e-12
	return float(vmin), float(vmax)


def plot_global_distribution(
	ax: plt.Axes,
	lon_grid: np.ndarray,
	lat_grid: np.ndarray,
	value_grid: np.ndarray,
	title: str,
	cmap: str,
	shading: str,
) -> None:
	vmin, vmax = percentile_limits(value_grid)

	mesh = ax.pcolormesh(
		lon_grid,
		lat_grid,
		value_grid,
		cmap=cmap,
		vmin=vmin,
		vmax=vmax,
		shading=shading,
		transform=ccrs.PlateCarree(),
		rasterized=True,
		zorder=1,
	)

	ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=2)
	ax.coastlines(linewidth=0.5, color="black", zorder=3)
	ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
	ax.set_xlabel("Longitude")
	ax.set_ylabel("Latitude")
	ax.set_title(title)
	gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", linestyle="--", alpha=0.6)
	gl.top_labels = False
	gl.right_labels = False
	gl.xlabel_style = {"size": 8}
	gl.ylabel_style = {"size": 8}

	cbar = ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
	cbar.set_label(title)


def draw_regional_window_boundaries(ax: plt.Axes) -> None:
	for lon_min, lon_max, lat_min, lat_max in REGIONAL_WINDOW_SEGMENTS:
		x = [lon_min, lon_max, lon_max, lon_min, lon_min]
		y = [lat_min, lat_min, lat_max, lat_max, lat_min]
		ax.plot(x, y, color="cyan", linewidth=1.0, transform=ccrs.PlateCarree(), zorder=4)


def plot_binary_mask(
	ax: plt.Axes,
	lon_grid: np.ndarray,
	lat_grid: np.ndarray,
	mask_grid: np.ndarray,
	title: str,
	shading: str,
) -> None:
	mask_values = np.where(mask_grid, 1.0, np.nan)
	mesh = ax.pcolormesh(
		lon_grid,
		lat_grid,
		mask_values,
		cmap="Blues",
		vmin=0.0,
		vmax=1.0,
		shading=shading,
		transform=ccrs.PlateCarree(),
		rasterized=True,
		zorder=1,
	)

	ax.add_feature(cfeature.LAND, facecolor="white", edgecolor="none", zorder=2)
	ax.coastlines(linewidth=0.5, color="black", zorder=3)
	ax.set_extent([-180, 180, -90, 90], crs=ccrs.PlateCarree())
	ax.set_xlabel("Longitude")
	ax.set_ylabel("Latitude")
	ax.set_title(title)
	gl = ax.gridlines(draw_labels=True, linewidth=0.5, color="gray", linestyle="--", alpha=0.6)
	gl.top_labels = False
	gl.right_labels = False
	gl.xlabel_style = {"size": 8}
	gl.ylabel_style = {"size": 8}

	cbar = ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
	cbar.set_ticks([1.0])
	cbar.set_ticklabels(["Kept"])
	cbar.set_label("Reusable Offshore Ocean Mask")


def plot_value_density(ax: plt.Axes, values: np.ndarray, value_column: str) -> None:
	finite_values = values[np.isfinite(values)]
	if finite_values.size == 0:
		raise ValueError("No finite values available for the density plot.")

	ax.hist(
		finite_values,
		bins=50,
		range=(0.0, 2.0),
		density=True,
		color="steelblue",
		alpha=0.75,
		edgecolor="white",
	)
	ax.set_xlim([0, 2])
	ax.set_xlabel(value_column)
	ax.set_ylabel("Probability Density")
	ax.set_title(f"{value_column} PDF After 1deg Downscaling and Before Screening")
	ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)


def aggregate_to_1deg_grid(df: pd.DataFrame, value_column: str) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	# Build 1-degree bins as [x, x+1), with centers at x+0.5.
	work = df.copy()
	work["lon_bin"] = np.floor(work["lon"]).astype(int)
	work["lat_bin"] = np.floor(work["lat"]).astype(int)

	grouped = work.groupby(["lat_bin", "lon_bin"], as_index=False)[value_column].mean()

	lon_centers = np.arange(-179.5, 180.0, 1.0)
	lat_centers = np.arange(-89.5, 90.0, 1.0)
	value_grid = np.full((lat_centers.size, lon_centers.size), np.nan, dtype=float)

	lon_idx = grouped["lon_bin"].to_numpy(dtype=int) + 180
	lat_idx = grouped["lat_bin"].to_numpy(dtype=int) + 90
	valid = (
		(lon_idx >= 0)
		& (lon_idx < lon_centers.size)
		& (lat_idx >= 0)
		& (lat_idx < lat_centers.size)
	)
	value_grid[lat_idx[valid], lon_idx[valid]] = grouped.loc[valid, value_column].to_numpy(dtype=float)

	lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
	return lon_grid, lat_grid, value_grid


def build_land_mask(lat_grid: np.ndarray, lon_grid: np.ndarray) -> np.ndarray:
	land_shp = shapereader.natural_earth(resolution="110m", category="physical", name="land")
	land_polygons = list(shapereader.Reader(land_shp).geometries())
	prepared_land = prep(unary_union(land_polygons))

	flat_lat = lat_grid.ravel()
	flat_lon = lon_grid.ravel()
	land_flat = np.array(
		[prepared_land.contains(Point(lon, lat)) for lat, lon in zip(flat_lat, flat_lon)],
		dtype=bool,
	)
	return land_flat.reshape(lat_grid.shape)


def latlon_to_unit_xyz(lat_deg: np.ndarray, lon_deg: np.ndarray) -> np.ndarray:
	lat_rad = np.deg2rad(lat_deg)
	lon_rad = np.deg2rad(lon_deg)
	x = np.cos(lat_rad) * np.cos(lon_rad)
	y = np.cos(lat_rad) * np.sin(lon_rad)
	z = np.sin(lat_rad)
	return np.column_stack((x, y, z))


def distance_km_to_land(lat_grid: np.ndarray, lon_grid: np.ndarray, land_mask: np.ndarray) -> np.ndarray:
	land_lat = lat_grid[land_mask]
	land_lon = lon_grid[land_mask]
	if land_lat.size == 0:
		return np.full(lat_grid.shape, np.inf, dtype=float)

	land_xyz = latlon_to_unit_xyz(land_lat, land_lon)
	tree = cKDTree(land_xyz)

	all_xyz = latlon_to_unit_xyz(lat_grid.ravel(), lon_grid.ravel())
	chord_dist, _ = tree.query(all_xyz, k=1)
	angles = 2.0 * np.arcsin(np.clip(chord_dist / 2.0, 0.0, 1.0))
	distance_km = 6371.0 * angles
	return distance_km.reshape(lat_grid.shape)


def regional_window_mask(lon_grid: np.ndarray, lat_grid: np.ndarray) -> np.ndarray:
	# Keep this mask definition consistent with REGIONAL_WINDOW_SEGMENTS.
	mask = np.zeros_like(lon_grid, dtype=bool)
	for lon_min, lon_max, lat_min, lat_max in REGIONAL_WINDOW_SEGMENTS:
		mask |= (
			(lon_grid >= lon_min)
			& (lon_grid <= lon_max)
			& (lat_grid >= lat_min)
			& (lat_grid <= lat_max)
		)
	return mask


def remove_small_connected_components(mask: np.ndarray, min_size: int = 4) -> np.ndarray:
	nrows, ncols = mask.shape
	visited = np.zeros_like(mask, dtype=bool)
	kept = np.zeros_like(mask, dtype=bool)

	neighbors = [
		(-1, -1), (-1, 0), (-1, 1),
		(0, -1),           (0, 1),
		(1, -1),  (1, 0),  (1, 1),
	]

	for r in range(nrows):
		for c in range(ncols):
			if visited[r, c] or not mask[r, c]:
				continue

			stack = [(r, c)]
			visited[r, c] = True
			component = []

			while stack:
				cr, cc = stack.pop()
				component.append((cr, cc))
				for dr, dc in neighbors:
					nr = cr + dr
					if nr < 0 or nr >= nrows:
						continue
					nc = (cc + dc) % ncols
					if visited[nr, nc] or not mask[nr, nc]:
						continue
					visited[nr, nc] = True
					stack.append((nr, nc))

			if len(component) >= min_size:
				for rr, cc in component:
					kept[rr, cc] = True

	return kept


def main() -> None:
	args = parse_args()
	csv_path = Path(args.csv_file)
	value_column = args.value_column
	default_year = 2021

	if not csv_path.exists():
		raise FileNotFoundError(f"CSV file not found: {csv_path}")

	df = pd.read_csv(csv_path)
	required_columns = ["lon", "lat", value_column]
	missing_columns = [col for col in required_columns if col not in df.columns]
	if missing_columns:
		raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

	subset = df[required_columns].replace([np.inf, -np.inf], np.nan).dropna()
	if subset.empty:
		raise ValueError("No valid rows found after removing NaN/Inf values.")

	lon_grid, lat_grid, value_grid = aggregate_to_1deg_grid(subset, value_column)
	land_mask = build_land_mask(lat_grid, lon_grid)
	dist_to_land_km = distance_km_to_land(lat_grid, lon_grid, land_mask)
	window_mask = regional_window_mask(lon_grid, lat_grid)
	offshore_ocean_mask = (~land_mask) & (dist_to_land_km >= 200.0) & window_mask
	offshore_ocean_mask_no_window = (~land_mask) & (dist_to_land_km >= 300.0)

	base_mask = np.isfinite(value_grid) & (value_grid >= 0.1) & offshore_ocean_mask_no_window
	final_mask = remove_small_connected_components(base_mask, min_size=3)
	value_filtered = np.where(final_mask, value_grid, np.nan)

	print(f"Cells with {value_column} data: {np.isfinite(value_grid).sum()}")
	print(f"Cells inside regional window: {window_mask.sum()}")
	print(f"Cells after {value_column}/distance filter: {base_mask.sum()}")
	print(f"Cells after component-size filter: {final_mask.sum()}")

	mask_output_path = Path(args.mask_output)
	np.savez_compressed(
		mask_output_path,
		offshore_ocean_mask=offshore_ocean_mask,
		lon_grid=lon_grid,
		lat_grid=lat_grid,
	)
	print(f"Saved reusable offshore ocean mask to: {mask_output_path.resolve()}")

	pdf_output_path = Path(args.pdf_output) if args.pdf_output else Path(f"figs/{value_column.lower()}_pdf_{default_year}.png")
	fig_pdf, ax_pdf = plt.subplots(figsize=(8, 5), dpi=180, constrained_layout=True)
	plot_value_density(ax_pdf, value_grid, value_column)
	fig_pdf.savefig(pdf_output_path, dpi=300)
	print(f"Saved pre-screening density plot to: {pdf_output_path.resolve()}")

	mask_plot_output_path = Path(args.mask_plot_output)
	fig_mask = plt.figure(figsize=(10, 5), dpi=180, constrained_layout=True)
	ax_mask = fig_mask.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
	plot_binary_mask(
		ax_mask,
		lon_grid,
		lat_grid,
		offshore_ocean_mask,
		"Reusable Offshore Ocean Mask (>300 km, regional window)",
		args.shading,
	)
	fig_mask.savefig(mask_plot_output_path, dpi=300)
	print(f"Saved reusable offshore ocean mask plot to: {mask_plot_output_path.resolve()}")

	fig = plt.figure(figsize=(10, 5), dpi=180, constrained_layout=True)
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
	plot_global_distribution(
		ax,
		lon_grid,
		lat_grid,
		value_filtered,
		f"{value_column} Ocean Distribution (1deg, filtered)",
		"inferno",
		args.shading,
	)
	draw_regional_window_boundaries(ax)

	output_path = Path(args.output)
	fig.savefig(output_path, dpi=300)
	print(f"Saved figure to: {output_path.resolve()}")


if __name__ == "__main__":
	main()
