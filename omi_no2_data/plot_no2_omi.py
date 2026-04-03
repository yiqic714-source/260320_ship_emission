#!/usr/bin/env python3
"""Plot OMI L3 global NO2 distribution from one HE5 file."""

from __future__ import annotations

from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
from netCDF4 import Dataset

try:
	import cartopy.crs as ccrs
	import cartopy.feature as cfeature
	import cartopy.io.shapereader as shpreader
	from shapely.ops import unary_union

	HAS_CARTOPY = True
except ImportError:
	HAS_CARTOPY = False


DATA_PATH = Path(
	"/home/chenyiqi/260320_ship_emission/"
	"OMI-Aura_L3-OMNO2d_2020m0509_v004-2025m1030t190837.he5"
)
VARIABLE_NAME = "ColumnAmountNO2"
OUTPUT_PATH = Path(f"omi_{VARIABLE_NAME}_global0509.png")


def _build_lon_lat_centers(lon_count: int, lat_count: int) -> tuple[np.ndarray, np.ndarray]:
	"""Build grid-cell center coordinates from known global span and grid size."""
	lon_edges = np.linspace(-180.0, 180.0, lon_count + 1)
	lat_edges = np.linspace(-90.0, 90.0, lat_count + 1)
	lons = 0.5 * (lon_edges[:-1] + lon_edges[1:])
	lats = 0.5 * (lat_edges[:-1] + lat_edges[1:])
	return lons, lats


def _read_omi_no2(file_path: Path, variable_name: str) -> tuple[np.ndarray, np.ndarray, np.ndarray, str]:
	"""Read selected NO2 field and return lons, lats, data, and unit string."""
	with Dataset(file_path, mode="r") as ds:
		data_fields = ds.groups["HDFEOS"].groups["GRIDS"].groups["ColumnAmountNO2"].groups[
			"Data Fields"
		]
		if variable_name not in data_fields.variables:
			raise KeyError(f"Variable {variable_name} not found in Data Fields.")

		var = data_fields.variables[variable_name]
		data = np.asarray(var[:], dtype=float)

		fill_candidates = []
		for attr in ("_FillValue", "MissingValue"):
			if hasattr(var, attr):
				fill_candidates.append(float(getattr(var, attr)))

		for fill_value in fill_candidates:
			data = np.where(np.isclose(data, fill_value), np.nan, data)

		scale = float(getattr(var, "ScaleFactor", 1.0))
		offset = float(getattr(var, "Offset", 0.0))
		data = data * scale + offset

		units = str(getattr(var, "Units", ""))
		n_lat, n_lon = data.shape
		lons, lats = _build_lon_lat_centers(n_lon, n_lat)

	return lons, lats, data, units


def _mask_land_data(lons: np.ndarray, lats: np.ndarray, data: np.ndarray) -> np.ndarray:
	"""Mask land pixels as NaN using Natural Earth land polygons."""
	land_shp = shpreader.natural_earth(resolution="110m", category="physical", name="land")
	land_geom = unary_union(list(shpreader.Reader(land_shp).geometries()))
	lon2d, lat2d = np.meshgrid(lons, lats)

	try:
		from shapely import contains_xy

		on_land = contains_xy(land_geom, lon2d, lat2d)
	except ImportError:
		from shapely import vectorized as shp_vect

		on_land = shp_vect.contains(land_geom, lon2d, lat2d)

	return np.where(on_land, np.nan, data)


def main() -> None:
	if not DATA_PATH.exists():
		raise FileNotFoundError(f"Input file not found: {DATA_PATH}")

	lons, lats, data, units = _read_omi_no2(DATA_PATH, VARIABLE_NAME)
	if HAS_CARTOPY:
		data = _mask_land_data(lons, lats, data)

	finite = np.isfinite(data)
	if not finite.any():
		raise ValueError("No valid values found after fill-value masking.")

	vmin = np.nanpercentile(data, 2)
	vmax = np.nanpercentile(data, 98)

	fig = plt.figure(figsize=(12, 5.8))

	if HAS_CARTOPY:
		ax = plt.axes(projection=ccrs.PlateCarree())
		ax.set_global()
		ax.coastlines(linewidth=0.7)
		ax.add_feature(cfeature.BORDERS, linewidth=0.3)
		mesh = ax.pcolormesh(
			lons,
			lats,
			data,
			transform=ccrs.PlateCarree(),
			shading="auto",
			cmap="viridis",
			vmin=vmin,
			vmax=vmax,
		)
	else:
		ax = plt.axes()
		mesh = ax.pcolormesh(
			lons,
			lats,
			data,
			shading="auto",
			cmap="viridis",
			vmin=vmin,
			vmax=vmax,
		)
		ax.set_xlabel("Longitude")
		ax.set_ylabel("Latitude")

	cbar = plt.colorbar(mesh, ax=ax, shrink=0.82, pad=0.04)
	if units:
		cbar.set_label(f"{VARIABLE_NAME} ({units})")
	else:
		cbar.set_label(VARIABLE_NAME)

	ax.set_title(f"OMI L3 {VARIABLE_NAME} global distribution")

	OUTPUT_PATH.parent.mkdir(parents=True, exist_ok=True)
	plt.tight_layout()
	plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
	plt.close(fig)

	print(f"Input file: {DATA_PATH}")
	print(f"Variable: {VARIABLE_NAME}")
	print(f"Output figure: {OUTPUT_PATH.resolve()}")


if __name__ == "__main__":
	main()
