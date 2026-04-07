import datetime as dt
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyhdf.SD import SD, SDC
from util import lon_to_utc_hour, read_and_mask_mod_variable


PROCESSED_DATA_DIR = Path('/home/chenyiqi/260320_ship_emission/processed_data')
MOD08_DIR = Path('/data/MODIS/MxD08_D3')
ERA5_ROOT = Path('/home/chenyiqi/260320_ship_emission/era5_daily_satllite_overpass_time')
SATELLITE_NAME = 'Aqua'  # 'Aqua' or 'Terra'
SL_SUFFIXES = ('oper_instant', 'oper_accum', 'wave_instant')


def _satellite_config(satellite_name: str) -> tuple[float, str, str, Path]:
	name = satellite_name.strip().lower()
	if name == 'aqua':
		return 13.5, '1330', 'MYD08'
	if name == 'terra':
		return 10.5, '1030', 'MOD08'
	raise ValueError(f'Unsupported SATELLITE_NAME: {satellite_name}')


TARGET_LST_HOUR, LST_TAG, MOD_PREFIX = _satellite_config(SATELLITE_NAME)
TARGET_LOCAL_HOUR = TARGET_LST_HOUR

MOD_VARS = {
	'Cloud_Retrieval_Fraction_Liquid': 'cf_ret_liq_mod08',
	'Cloud_Optical_Thickness_Liquid_Mean': 'cot_mod08',
	'Cloud_Water_Path_Liquid_Mean': 'cwp_mod08',
	'Cloud_Effective_Radius_Liquid_Mean': 'cer_mod08',
	'Cloud_Retrieval_Fraction_Combined': 'cf_ret_combined_mod08',
	'Aerosol_Optical_Depth_Land_Ocean_Mean': 'aod_mod08',
	'Solar_Zenith_Mean': 'sza_mod08',
	'Solar_Azimuth_Mean': 'saa_mod08',
}

def parse_target_date_from_argv(argv: list[str]) -> dt.date:
	if len(argv) != 4:
		raise SystemExit('Usage: python ml_xy_preparation.py yyyy mm dd')
	try:
		return dt.date(int(argv[1]), int(argv[2]), int(argv[3]))
	except ValueError as exc:
		raise SystemExit(f'Invalid date arguments: {exc}') from exc


def find_mod08_file_for_date(target_date: dt.date, mod08_dir: Path) -> Path:
	yyyyddd = target_date.strftime('%Y%j')
	pattern = str(mod08_dir / f'{MOD_PREFIX}_D3.A{yyyyddd}.061.*.hdf')
	matches = sorted(glob.glob(pattern))
	if not matches:
		raise FileNotFoundError(f'No {MOD_PREFIX} file matched: {pattern}')
	return Path(matches[0])


def find_soxdiff_track_csv(target_date: dt.date) -> Path:
	path = PROCESSED_DATA_DIR / f'hysplit_rsl/{target_date:%Y}/soxdiff_track_mean_{target_date:%Y%m%d}{LST_TAG}.csv'
	if not path.exists():
		raise FileNotFoundError(f'Cannot find soxdiff track CSV: {path}')
	return path


def load_mod08_data(mod_file: Path) -> dict:
	hdf = SD(str(mod_file), SDC.READ)
	try:
		lon = hdf.select('XDim')[:]
		lat = hdf.select('YDim')[:]
		var_data = {out_name: read_and_mask_mod_variable(hdf, in_name) for in_name, out_name in MOD_VARS.items()}
	finally:
		hdf.end()

	return {
		'lat': lat,
		'lon': lon,
		'vars': var_data,
	}


def _nearest_grid_values(grid_values: np.ndarray, point_values: np.ndarray) -> np.ndarray:
	grid = np.asarray(grid_values, dtype=float)
	points = np.asarray(point_values, dtype=float)
	if grid.ndim != 1 or grid.size == 0:
		raise ValueError('grid_values must be a non-empty 1D array.')

	reversed_order = bool(grid[0] > grid[-1])
	if reversed_order:
		grid_sorted = grid[::-1]
	else:
		grid_sorted = grid

	idx = np.searchsorted(grid_sorted, points)
	idx = np.clip(idx, 1, grid_sorted.size - 1)
	left = grid_sorted[idx - 1]
	right = grid_sorted[idx]
	nearest_idx = np.where(np.abs(points - left) <= np.abs(right - points), idx - 1, idx)
	return grid_sorted[nearest_idx]


def attach_mod_values(point_df: pd.DataFrame, mod_data: dict, lat_col: str, lon_col: str) -> pd.DataFrame:
	out_df = point_df.copy()
	lat_grid = np.asarray(mod_data['lat'], dtype=float)
	lon_grid = np.asarray(mod_data['lon'], dtype=float)

	lon2d, lat2d = np.meshgrid(lon_grid, lat_grid)
	mod_flat = {
		'mod_lat': lat2d.ravel(),
		'mod_lon': lon2d.ravel(),
	}
	for out_name, data_2d in mod_data['vars'].items():
		mod_flat[out_name] = np.asarray(data_2d, dtype=float).ravel()
	mod_df = pd.DataFrame(mod_flat)

	out_df['mod_lat'] = _nearest_grid_values(lat_grid, out_df[lat_col].to_numpy())
	out_df['mod_lon'] = _nearest_grid_values(lon_grid, out_df[lon_col].to_numpy())
	out_df = out_df.merge(mod_df, on=['mod_lat', 'mod_lon'], how='left')
	out_df = out_df.drop(columns=['mod_lat', 'mod_lon'])
	return out_df


def _to_dataset_lon_value(ds_lon: np.ndarray, point_lon: float) -> float:
	if np.nanmax(ds_lon) > 180 and point_lon < 0:
		return point_lon + 360.0
	if np.nanmax(ds_lon) <= 180 and point_lon > 180:
		return point_lon - 360.0
	return point_lon


def _to_dataset_lon_values(ds_lon: np.ndarray, point_lons: np.ndarray) -> np.ndarray:
	point_lons = np.asarray(point_lons, dtype=float)
	if np.nanmax(ds_lon) > 180:
		return np.where(point_lons < 0, point_lons + 360.0, point_lons)
	return np.where(point_lons > 180, point_lons - 360.0, point_lons)


def _nearest_index(values: np.ndarray, target: float) -> int:
	return int(np.abs(values - target).argmin())


def _nearest_indices(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
	values = np.asarray(values, dtype=float)
	targets = np.asarray(targets, dtype=float)
	if values.ndim != 1 or values.size == 0:
		raise ValueError('values must be a non-empty 1D array.')

	reversed_order = bool(values[0] > values[-1])
	if reversed_order:
		vals = values[::-1]
	else:
		vals = values

	idx = np.searchsorted(vals, targets)
	idx = np.clip(idx, 1, vals.size - 1)
	left = vals[idx - 1]
	right = vals[idx]
	nearest = np.where(np.abs(targets - left) <= np.abs(right - targets), idx - 1, idx)

	if reversed_order:
		return (values.size - 1 - nearest).astype(int)
	return nearest.astype(int)


def _find_time_index(ds: xr.Dataset, target_date: dt.date) -> int:
	time_values = pd.to_datetime(ds['valid_time'].values)
	date_values = time_values.date
	matches = np.where(date_values == target_date)[0]
	if len(matches) == 0:
		raise ValueError(f'Cannot find date {target_date} in valid_time.')
	return int(matches[0])


def _build_era5_paths(target_date: dt.date, utc_hour: int) -> dict[str, Path]:
	year = target_date.year
	yyyymm = target_date.strftime('%Y%m')
	hh = f'{utc_hour:02d}'
	pl_path = ERA5_ROOT / f'{year}_LST{LST_TAG}_pl' / f'era5_pl_{yyyymm}_utc{hh}.nc'
	sl_paths = {
		suffix: ERA5_ROOT / f'{year}_LST{LST_TAG}_sl' / f'era5_sl_{yyyymm}_utc{hh}_{suffix}.nc'
		for suffix in SL_SUFFIXES
	}
	paths = {'pl': pl_path, **sl_paths}
	for tag, path in paths.items():
		if not path.exists():
			raise FileNotFoundError(f'ERA5 file missing for {tag}: {path}')
	return paths


def _extract_era5_for_points(
	ds: xr.Dataset,
	time_idx: int,
	lat_idx: np.ndarray,
	lon_idx: np.ndarray,
	source_tag: str,
) -> dict[str, np.ndarray]:
	features: dict[str, np.ndarray] = {}
	for var_name, da in ds.data_vars.items():
		dims = set(da.dims)
		if 'valid_time' not in dims or 'latitude' not in dims or 'longitude' not in dims:
			continue

		if 'pressure_level' in dims:
			arr = np.asarray(da.isel(valid_time=time_idx).values)
			# arr shape: [pressure_level, latitude, longitude]
			values_2d = arr[:, lat_idx, lon_idx]
			levels = np.asarray(ds['pressure_level'].values)
			for k, level in enumerate(levels):
				level_int = int(round(float(level)))
				features[f'{var_name}_{level_int}'] = values_2d[k, :].astype(float)
		else:
			arr = np.asarray(da.isel(valid_time=time_idx).values)
			col_name = var_name if var_name not in features else f'{var_name}_{source_tag}'
			features[col_name] = arr[lat_idx, lon_idx].astype(float)
	return features


def extract_era5_for_t0_points(mod_df: pd.DataFrame, target_date: dt.date) -> pd.DataFrame:
	if mod_df.empty:
		return mod_df.copy()
	required_cols = {'t0_lat', 't0_lon'}
	if not required_cols.issubset(set(mod_df.columns)):
		raise ValueError('Input CSV must contain t0_lat and t0_lon columns.')

	ds_cache: dict[str, xr.Dataset] = {}
	out_df = mod_df.reset_index(drop=True).copy()
	out_df['_row_id'] = np.arange(len(out_df), dtype=int)
	out_df['_utc_hour'] = np.array(
		[lon_to_utc_hour(float(lon), target_lst_hour=TARGET_LST_HOUR) for lon in out_df['t0_lon'].to_numpy()],
		dtype=int,
	)
	era_chunks: list[pd.DataFrame] = []

	try:
		for utc_hour, group in out_df.groupby('_utc_hour', sort=False):
			paths = _build_era5_paths(target_date, int(utc_hour))
			group_rows = group['_row_id'].to_numpy(dtype=int)
			group_lats = group['t0_lat'].to_numpy(dtype=float)
			group_lons = group['t0_lon'].to_numpy(dtype=float)
			feature_block: dict[str, np.ndarray] = {'_row_id': group_rows}

			for tag, path in paths.items():
				path_key = str(path)
				if path_key not in ds_cache:
					ds_cache[path_key] = xr.open_dataset(path)
				ds = ds_cache[path_key]

				time_idx = _find_time_index(ds, target_date)
				lat_values = np.asarray(ds['latitude'].values, dtype=float)
				lon_values = np.asarray(ds['longitude'].values, dtype=float)
				lon_for_ds = _to_dataset_lon_values(lon_values, group_lons)
				lat_idx = _nearest_indices(lat_values, group_lats)
				lon_idx = _nearest_indices(lon_values, lon_for_ds)

				feature_block.update(_extract_era5_for_points(ds, time_idx, lat_idx, lon_idx, source_tag=tag))

			era_chunks.append(pd.DataFrame(feature_block))
	finally:
		for ds in ds_cache.values():
			ds.close()

	era_df = pd.concat(era_chunks, axis=0, ignore_index=True) if era_chunks else pd.DataFrame({'_row_id': []})
	result_df = out_df.merge(era_df, on='_row_id', how='left').drop(columns=['_row_id', '_utc_hour'])
	return result_df


if __name__ == "__main__":
	target_date = parse_target_date_from_argv(sys.argv)
	input_csv = find_soxdiff_track_csv(target_date)
	output_csv = PROCESSED_DATA_DIR / f'ml_xy_data0407/{target_date:%Y}/soxdiff_met_and_cld_{target_date:%Y%m%d}{LST_TAG}.csv'

	mod_file = find_mod08_file_for_date(target_date, MOD08_DIR)
	affected_df = pd.read_csv(input_csv)
	mod_data = load_mod08_data(mod_file)
	row_with_vars = attach_mod_values(affected_df, mod_data, lat_col='t0_lat', lon_col='t0_lon')
	result_df = extract_era5_for_t0_points(row_with_vars, target_date)

	output_csv.parent.mkdir(parents=True, exist_ok=True)
	result_df.to_csv(output_csv, index=False)

	# print(f'SATELLITE_NAME: {SATELLITE_NAME}')
	# print(f'TARGET_LST_HOUR: {TARGET_LST_HOUR}')
	print(f'MOD file: {mod_file}')
	print(f'Input CSV: {input_csv}')
	print(f'Output rows: {len(result_df)} / {len(affected_df)}')
	print(f'Saved CSV: {output_csv}')
