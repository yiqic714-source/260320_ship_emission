#!/usr/bin/env python3
"""Plot a MERRA-2 variable at the time nearest to 13:00 local time."""

from __future__ import annotations

import os
from pathlib import Path

import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
import xarray as xr

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    import cartopy.io.shapereader as shpreader
    from shapely.ops import unary_union

    try:
        from shapely import contains_xy
    except ImportError:
        contains_xy = None

    try:
        from shapely import vectorized as shapely_vectorized
    except ImportError:
        shapely_vectorized = None

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


DATA_PATH = Path(
    "/data/chenyiqi/merra2/2019_2020_sox_tau/"
    "M2T1NXAER.5.12.4_MERRA2_400.tavg1_2d_aer_Nx.20190101_subsetted.nc4"
)
VARIABLE_NAME = "SO2SMASS"#"TOTEXTTAU"
OUTPUT_PATH = Path(f"figs/merra2_{VARIABLE_NAME}_distribution.png")
TARGET_HOUR = 13
TARGET_MINUTE = 0


def _build_ocean_mask(lons: np.ndarray, lats: np.ndarray) -> np.ndarray:
    """Return True over ocean and False over land."""
    land_shp = shpreader.natural_earth(
        resolution="110m", category="physical", name="land"
    )
    land_geom = unary_union(list(shpreader.Reader(land_shp).geometries()))

    lon2d, lat2d = np.meshgrid(lons, lats)
    if contains_xy is not None:
        on_land = contains_xy(land_geom, lon2d, lat2d)
    elif shapely_vectorized is not None:
        on_land = shapely_vectorized.contains(land_geom, lon2d, lat2d)
    else:
        raise ImportError("shapely vectorized point-in-polygon support is required.")

    return ~on_land


def _find_nearest_13lt_indices(time_da: xr.DataArray, lons: np.ndarray) -> np.ndarray:
    """Return one time index per longitude for the slice nearest to 13:00 local time."""
    if np.issubdtype(time_da.dtype, np.datetime64):
        time_values = pd.to_datetime(time_da.values)
        utc_hours = (
            time_values.hour
            + time_values.minute / 60.0
            + time_values.second / 3600.0
        ).to_numpy()
        local_hours = (utc_hours[:, None] + lons[None, :] / 15.0) % 24.0
        target_hour = TARGET_HOUR + TARGET_MINUTE / 60.0
        hour_diff = np.abs(local_hours - target_hour)
        circular_diff = np.minimum(hour_diff, 24.0 - hour_diff)
        return np.argmin(circular_diff, axis=0)

    # Fallback for non-decoded numeric time: infer minutes from time units.
    units = str(time_da.attrs.get("units", "")).lower()
    vals = np.asarray(time_da.values, dtype=float)
    if "minutes since" in units:
        utc_hours = ((vals + 30.0) / 60.0) % 24.0
        local_hours = (utc_hours[:, None] + lons[None, :] / 15.0) % 24.0
        target_hour = TARGET_HOUR + TARGET_MINUTE / 60.0
        hour_diff = np.abs(local_hours - target_hour)
        circular_diff = np.minimum(hour_diff, 24.0 - hour_diff)
        return np.argmin(circular_diff, axis=0)

    raise ValueError("Unsupported time format; please decode time to datetime first.")


def main() -> None:
    if not DATA_PATH.exists():
        raise FileNotFoundError(f"Input file not found: {DATA_PATH}")

    ds = xr.open_dataset(DATA_PATH, decode_times=True)

    if VARIABLE_NAME not in ds.variables:
        raise KeyError(f"Variable {VARIABLE_NAME} not found in dataset.")

    lats = ds["lat"].values
    lons = ds["lon"].values
    t_indices = _find_nearest_13lt_indices(ds["time"], lons)
    variable_data = np.asarray(ds[VARIABLE_NAME].values, dtype=float)
    selected_values = variable_data[t_indices, :, np.arange(len(lons))].T
    ocean_mask = _build_ocean_mask(lons, lats)

    plot_values = np.where(ocean_mask, selected_values, np.nan)
    plot_values = np.where(plot_values > 0, plot_values, np.nan)

    if not np.isfinite(plot_values).any():
        raise ValueError(f"No positive {VARIABLE_NAME} values remain after land masking.")

    vmin = np.nanmin(plot_values)
    vmax = np.nanmax(plot_values)
    norm = LogNorm(vmin=vmin, vmax=vmax)

    selected_utc_times = pd.to_datetime(ds["time"].values[t_indices])
    target_hour = TARGET_HOUR + TARGET_MINUTE / 60.0
    local_hour_values = (
        selected_utc_times.hour
        + selected_utc_times.minute / 60.0
        + selected_utc_times.second / 3600.0
        + lons / 15.0
    ) % 24.0
    local_hour_offset = np.abs(local_hour_values - target_hour)
    local_hour_offset = np.minimum(local_hour_offset, 24.0 - local_hour_offset)

    fig = plt.figure(figsize=(12, 5.8))

    if HAS_CARTOPY:
        ax = plt.axes(projection=ccrs.PlateCarree())
        ax.set_global()
        ax.coastlines(linewidth=0.7)
        ax.add_feature(cfeature.BORDERS, linewidth=0.3)

        if len(lons) > 1 and len(lats) > 1:
            mesh = ax.pcolormesh(
                lons,
                lats,
                plot_values,
                transform=ccrs.PlateCarree(),
                shading="auto",
                cmap="YlOrRd",
                norm=norm,
            )
        else:
            lon2d, lat2d = np.meshgrid(lons, lats)
            ocean_values = plot_values.ravel()
            valid = np.isfinite(ocean_values)
            mesh = ax.scatter(
                lon2d.ravel()[valid],
                lat2d.ravel()[valid],
                c=ocean_values[valid],
                s=12,
                cmap="YlOrRd",
                norm=norm,
                transform=ccrs.PlateCarree(),
            )
    else:
        ax = plt.axes()
        if len(lons) > 1 and len(lats) > 1:
            mesh = ax.pcolormesh(
                lons,
                lats,
                plot_values,
                shading="auto",
                cmap="YlOrRd",
                norm=norm,
            )
        else:
            lon2d, lat2d = np.meshgrid(lons, lats)
            ocean_values = plot_values.ravel()
            valid = np.isfinite(ocean_values)
            mesh = ax.scatter(
                lon2d.ravel()[valid],
                lat2d.ravel()[valid],
                c=ocean_values[valid],
                s=12,
                cmap="YlOrRd",
                norm=norm,
            )
        ax.set_xlabel("Longitude")
        ax.set_ylabel("Latitude")

    cbar = plt.colorbar(mesh, ax=ax, shrink=0.82, pad=0.04)
    cbar.set_label(f"{VARIABLE_NAME} (kg m-2)")

    ax.set_title(
        f"MERRA-2 {VARIABLE_NAME} nearest 13:00 local time"
    )

    plt.tight_layout()
    plt.savefig(OUTPUT_PATH, dpi=300, bbox_inches="tight")
    plt.close(fig)

    print(f"Selected UTC indices by longitude: {t_indices.tolist()}")
    print(
        "Selected UTC time range: "
        f"{selected_utc_times.min().strftime('%Y-%m-%d %H:%M UTC')} to "
        f"{selected_utc_times.max().strftime('%Y-%m-%d %H:%M UTC')}"
    )
    print(
        "Local-time offset from 13:00 range (hours): "
        f"{local_hour_offset.min():.2f} to {local_hour_offset.max():.2f}"
    )
    print(f"Output figure: {os.path.abspath(OUTPUT_PATH)}")


if __name__ == "__main__":
    main()
