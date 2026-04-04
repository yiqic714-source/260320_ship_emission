import datetime as dt
import glob
import sys
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from pyhdf.SD import SD, SDC
from util import lon_to_utc_hour


PROCESSED_DATA_DIR = Path('/home/chenyiqi/260320_ship_emission/processed_data')
MOD08_DIR = Path('/data/MODIS/MxD08_D3')
ERA5_ROOT_BASE = Path('/home/chenyiqi/260320_ship_emission')
SATELLITE_NAME = 'Aqua'  # 'Aqua' or 'Terra'
SL_SUFFIXES = ('oper_instant', 'oper_accum', 'wave_instant')


def _satellite_config(satellite_name: str) -> tuple[float, str, str, Path]:
	name = satellite_name.strip().lower()
	if name == 'aqua':
		return 13.5, '1330', 'MYD08', ERA5_ROOT_BASE / 'era5_daily_Aqua_time'
	if name == 'terra':
		return 10.5, '1030', 'MOD08', ERA5_ROOT_BASE / 'era5_daily_Terra_time'
	raise ValueError(f'Unsupported SATELLITE_NAME: {satellite_name}')


TARGET_LST_HOUR, LST_TAG, MOD_PREFIX, ERA5_ROOT = _satellite_config(SATELLITE_NAME)
TARGET_LOCAL_HOUR = TARGET_LST_HOUR

MOD_VARS = {
	'Cloud_Retrieval_Fraction_Liquid': 'cf_ret_liq_mod08',
	'Cloud_Optical_Thickness_Liquid_Mean': 'cot_mod08',
	'Cloud_Effective_Radius_Liquid_Mean': 'cer_mod08',
	'Solar_Zenith_Mean': 'sza_mod08',
	'Cloud_Retrieval_Fraction_Combined': 'cf_ret_combined_mod08',
	'Solar_Azimuth_Mean': 'saa_mod08',
}


def read_and_mask_mod_variable(hdf, var_name):
	"""Read HDF variable and apply fill/offset/scale in the same style as reference code."""
	sds = hdf.select(var_name)
	data = sds[:].astype(float)
	attrs = sds.attributes()
	fill_value = attrs.get('_FillValue', None)
	scale_factor = attrs.get('scale_factor')
	offset = attrs.get('add_offset')
	if fill_value is not None:
		data[data == fill_value] = float('nan')
	if offset is not None:
		data = data - offset
	if scale_factor is not None:
		data = data * scale_factor
	return data


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
	path = PROCESSED_DATA_DIR / f'hysplit_rsl/soxdiff_track_mean_{target_date:%Y%m%d}{LST_TAG}.csv'
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


def attach_mod_values(point_df: pd.DataFrame, mod_data: dict, lat_col: str, lon_col: str) -> pd.DataFrame:
	out_df = point_df.copy()
	lat_grid = mod_data['lat']
	lon_grid = mod_data['lon']

	for out_name, data_2d in mod_data['vars'].items():
		values = []
		for lat_value, lon_value in zip(out_df[lat_col].to_numpy(), out_df[lon_col].to_numpy()):
			lat_idx = int(abs(lat_grid - lat_value).argmin())
			lon_idx = int(abs(lon_grid - lon_value).argmin())
			values.append(float(data_2d[lat_idx, lon_idx]))
		out_df[out_name] = values

	return out_df


def _to_dataset_lon_value(ds_lon: np.ndarray, point_lon: float) -> float:
	if np.nanmax(ds_lon) > 180 and point_lon < 0:
		return point_lon + 360.0
	if np.nanmax(ds_lon) <= 180 and point_lon > 180:
		return point_lon - 360.0
	return point_lon


def _nearest_index(values: np.ndarray, target: float) -> int:
	return int(np.abs(values - target).argmin())


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


def _extract_era5_from_dataset(
	ds: xr.Dataset,
	time_idx: int,
	lat_idx: int,
	lon_idx: int,
	source_tag: str,
) -> dict[str, float]:
	features: dict[str, float] = {}
	for var_name, da in ds.data_vars.items():
		dims = set(da.dims)
		if 'valid_time' not in dims or 'latitude' not in dims or 'longitude' not in dims:
			continue

		if 'pressure_level' in dims:
			values = da.isel(valid_time=time_idx, latitude=lat_idx, longitude=lon_idx).values
			levels = ds['pressure_level'].values
			for level, value in zip(levels, np.asarray(values)):
				level_int = int(round(float(level)))
				features[f'{var_name}_{level_int}'] = float(value)
		else:
			value = float(np.asarray(da.isel(valid_time=time_idx, latitude=lat_idx, longitude=lon_idx).values).squeeze())
			col_name = var_name if var_name not in features else f'{var_name}_{source_tag}'
			features[col_name] = value
	return features


def extract_era5_for_t0_points(mod_df: pd.DataFrame, target_date: dt.date) -> pd.DataFrame:
	if mod_df.empty:
		return mod_df.copy()
	required_cols = {'t0_lat', 't0_lon'}
	if not required_cols.issubset(set(mod_df.columns)):
		raise ValueError('Input CSV must contain t0_lat and t0_lon columns.')

	ds_cache: dict[str, xr.Dataset] = {}
	era_rows: list[dict[str, float]] = []

	try:
		for _, row in mod_df.iterrows():
			start_lat = float(row['t0_lat'])
			start_lon = float(row['t0_lon'])
			utc_hour = lon_to_utc_hour(start_lon, target_lst_hour=TARGET_LST_HOUR)
			paths = _build_era5_paths(target_date, utc_hour)

			feature_row: dict[str, float] = {}

			for tag, path in paths.items():
				path_key = str(path)
				if path_key not in ds_cache:
					ds_cache[path_key] = xr.open_dataset(path)
				ds = ds_cache[path_key]

				time_idx = _find_time_index(ds, target_date)
				lat_values = ds['latitude'].values
				lon_values = ds['longitude'].values
				lon_for_ds = _to_dataset_lon_value(lon_values, start_lon)
				lat_idx = _nearest_index(lat_values, start_lat)
				lon_idx = _nearest_index(lon_values, lon_for_ds)

				feature_row.update(_extract_era5_from_dataset(ds, time_idx, lat_idx, lon_idx, source_tag=tag))

			era_rows.append(feature_row)
	finally:
		for ds in ds_cache.values():
			ds.close()

	era_df = pd.DataFrame(era_rows)
	return pd.concat([mod_df.reset_index(drop=True), era_df], axis=1)


if __name__ == "__main__":
	target_date = parse_target_date_from_argv(sys.argv)
	input_csv = find_soxdiff_track_csv(target_date)
	output_csv = PROCESSED_DATA_DIR / f'ml_xy_data/soxdiff_met_and_cld_{target_date:%Y%m%d}.csv'

	mod_file = find_mod08_file_for_date(target_date, MOD08_DIR)
	affected_df = pd.read_csv(input_csv)
	mod_data = load_mod08_data(mod_file)
	row_with_vars = attach_mod_values(affected_df, mod_data, lat_col='t0_lat', lon_col='t0_lon')
	result_df = extract_era5_for_t0_points(row_with_vars, target_date)

	output_csv.parent.mkdir(parents=True, exist_ok=True)
	result_df.to_csv(output_csv, index=False)

	print(f'SATELLITE_NAME: {SATELLITE_NAME}')
	print(f'TARGET_LST_HOUR: {TARGET_LST_HOUR}')
	print(f'MOD file: {mod_file}')
	print(f'Input CSV: {input_csv}')
	print(f'Input rows: {len(affected_df)}')
	print(f'Output rows: {len(result_df)}')
	print(f'Saved CSV: {output_csv}')
