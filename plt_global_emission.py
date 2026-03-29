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


def parse_args() -> argparse.Namespace:
	parser = argparse.ArgumentParser(
		description="Plot global PM distribution over ocean on a 1-degree grid."
	)
	parser.add_argument(
		"csv_file",
		nargs="?",
		default="L01dA_2013_global.csv",
		help="Input CSV file path (default: L01dA_2013_global.csv)",
	)
	parser.add_argument(
		"--output",
		default="pm_ocean_global_distribution_1deg.png",
		help="Output image path (default: pm_ocean_global_distribution_1deg.png)",
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
	ax.grid(True, linestyle="--", linewidth=0.5, alpha=0.5)

	cbar = ax.figure.colorbar(mesh, ax=ax, fraction=0.046, pad=0.04)
	cbar.set_label(title)


def aggregate_to_1deg_grid(df: pd.DataFrame) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
	# Build 1-degree bins as [x, x+1), with centers at x+0.5.
	work = df.copy()
	work["lon_bin"] = np.floor(work["lon"]).astype(int)
	work["lat_bin"] = np.floor(work["lat"]).astype(int)

	grouped = work.groupby(["lat_bin", "lon_bin"], as_index=False)["PM"].mean()

	lon_centers = np.arange(-179.5, 180.0, 1.0)
	lat_centers = np.arange(-89.5, 90.0, 1.0)
	pm_grid = np.full((lat_centers.size, lon_centers.size), np.nan, dtype=float)

	lon_idx = grouped["lon_bin"].to_numpy(dtype=int) + 180
	lat_idx = grouped["lat_bin"].to_numpy(dtype=int) + 90
	valid = (
		(lon_idx >= 0)
		& (lon_idx < lon_centers.size)
		& (lat_idx >= 0)
		& (lat_idx < lat_centers.size)
	)
	pm_grid[lat_idx[valid], lon_idx[valid]] = grouped.loc[valid, "PM"].to_numpy(dtype=float)

	lon_grid, lat_grid = np.meshgrid(lon_centers, lat_centers)
	return lon_grid, lat_grid, pm_grid


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


def remove_small_connected_components(mask: np.ndarray, min_size: int = 6) -> np.ndarray:
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

	if not csv_path.exists():
		raise FileNotFoundError(f"CSV file not found: {csv_path}")

	df = pd.read_csv(csv_path)
	required_columns = ["lon", "lat", "PM"]
	missing_columns = [col for col in required_columns if col not in df.columns]
	if missing_columns:
		raise ValueError(f"Missing required columns: {', '.join(missing_columns)}")

	subset = df[required_columns].replace([np.inf, -np.inf], np.nan).dropna()
	if subset.empty:
		raise ValueError("No valid rows found after removing NaN/Inf values.")

	lon_grid, lat_grid, pm_grid = aggregate_to_1deg_grid(subset)
	land_mask = build_land_mask(lat_grid, lon_grid)
	dist_to_land_km = distance_km_to_land(lat_grid, lon_grid, land_mask)

	base_mask = np.isfinite(pm_grid) & (pm_grid >= 0.5) & (dist_to_land_km >= 300.0)
	final_mask = remove_small_connected_components(base_mask, min_size=6)
	pm_filtered = np.where(final_mask, pm_grid, np.nan)

	print(f"Cells with PM data: {np.isfinite(pm_grid).sum()}")
	print(f"Cells after PM/distance filter: {base_mask.sum()}")
	print(f"Cells after component-size filter: {final_mask.sum()}")

	fig = plt.figure(figsize=(10, 5), dpi=180, constrained_layout=True)
	ax = fig.add_subplot(1, 1, 1, projection=ccrs.PlateCarree())
	plot_global_distribution(ax, lon_grid, lat_grid, pm_filtered, "PM Ocean Distribution (1deg, filtered)", "inferno", args.shading)

	output_path = Path(args.output)
	fig.savefig(output_path, dpi=300)
	print(f"Saved figure to: {output_path.resolve()}")


if __name__ == "__main__":
	main()
