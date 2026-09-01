from __future__ import annotations

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
ACCU_SOX_DIR = PROCESSED_DATA_DIR / 'accu_sox_grid'
OUT_DIR = PROCESSED_DATA_DIR / 'ml_grid_data'
MOD08_DIR = Path('/data/MODIS/MxD08/MxD08_D3')
ERA5_ROOT = Path('/home/chenyiqi/260320_ship_emission/era5_daily_satllite_overpass_time')
SATELLITE_NAME = 'Aqua'  # 'Aqua' or 'Terra'
SL_SUFFIXES = ('oper_instant', 'oper_accum', 'wave_instant')
GAMM = 1.37e-5


def _satellite_config(satellite_name: str) -> tuple[float, str, str]:
    name = satellite_name.strip().lower()
    if name == 'aqua':
        return 13.5, '1330', 'MYD08'
    if name == 'terra':
        return 10.5, '1030', 'MOD08'
    raise ValueError(f'Unsupported SATELLITE_NAME: {satellite_name}')


TARGET_LST_HOUR, LST_TAG, MOD_PREFIX = _satellite_config(SATELLITE_NAME)

MOD_VARS = {
    'Cloud_Retrieval_Fraction_Liquid': 'cf_ret_liq_mod08',
    'Cloud_Optical_Thickness_Liquid_Mean': 'cot_mod08',
    'Cloud_Water_Path_Liquid_Mean': 'cwp_mod08',
    'Cloud_Effective_Radius_Liquid_Mean': 'cer_mod08',
    'Cloud_Retrieval_Fraction_Combined': 'cf_ret_combined_mod08',
    'Aerosol_Optical_Depth_Land_Ocean_Mean': 'aod_mod08',
    'Sensor_Zenith_Mean': 'ssza_mod08',
    'Sensor_Azimuth_Mean': 'ssaa_mod08',
    'Solar_Zenith_Mean': 'slza_mod08',
    'Solar_Azimuth_Mean': 'slaa_mod08',
}


def iter_dates(year_start: int, year_end: int) -> list[dt.date]:
    start = dt.date(year_start, 1, 1)
    end = dt.date(year_end, 12, 31)
    days = (end - start).days + 1
    return [start + dt.timedelta(days=i) for i in range(days)]


def parse_dates_from_argv(argv: list[str]) -> list[dt.date]:
    if len(argv) == 2:
        year = int(argv[1])
        return iter_dates(year, year)

    raise SystemExit('Usage: python xxx.py yyyy')


def find_mod08_file_for_date(target_date: dt.date, mod08_dir: Path) -> Path:
    date_tag = target_date.strftime('%Y%j')
    pattern = str(mod08_dir / f'{MOD_PREFIX}_D3.A{date_tag}.061.*.hdf')
    matches = sorted(glob.glob(pattern))
    if not matches:
        raise FileNotFoundError(f'No {MOD_PREFIX} file matched: {pattern}')
    return Path(matches[0])


def find_accu_sox_file(target_date: dt.date) -> Path:
    path = ACCU_SOX_DIR / f'{target_date:%Y}' / f'accu_sox_{target_date:%Y%m}_{LST_TAG}.nc'
    if not path.exists():
        raise FileNotFoundError(f'Cannot find monthly accu_sox grid: {path}')
    return path


def load_mod08_data(mod_file: Path) -> tuple[np.ndarray, np.ndarray, dict[str, np.ndarray]]:
    hdf = SD(str(mod_file), SDC.READ)
    try:
        lon = hdf.select('XDim')[:].astype(float)
        lat = hdf.select('YDim')[:].astype(float)
        var_data = {
            out_name: read_and_mask_mod_variable(hdf, in_name).astype(float)
            for in_name, out_name in MOD_VARS.items()
        }
    finally:
        hdf.end()
    return lat, lon, var_data


def _nearest_indices(values: np.ndarray, targets: np.ndarray) -> np.ndarray:
    values = np.asarray(values, dtype=float)
    targets = np.asarray(targets, dtype=float)
    reversed_order = bool(values[0] > values[-1])
    vals = values[::-1] if reversed_order else values
    idx = np.searchsorted(vals, targets)
    idx = np.clip(idx, 1, vals.size - 1)
    left = vals[idx - 1]
    right = vals[idx]
    nearest = np.where(np.abs(targets - left) <= np.abs(right - targets), idx - 1, idx)
    if reversed_order:
        return (values.size - 1 - nearest).astype(int)
    return nearest.astype(int)


def _to_dataset_lon_values(ds_lon: np.ndarray, point_lons: np.ndarray) -> np.ndarray:
    point_lons = np.asarray(point_lons, dtype=float)
    if np.nanmax(ds_lon) > 180:
        return np.where(point_lons < 0, point_lons + 360.0, point_lons)
    return np.where(point_lons > 180, point_lons - 360.0, point_lons)


def regrid_modis_to_target(
    mod_lat: np.ndarray,
    mod_lon: np.ndarray,
    mod_vars: dict[str, np.ndarray],
    target_lat: np.ndarray,
    target_lon: np.ndarray,
) -> dict[str, np.ndarray]:
    lat_idx = _nearest_indices(mod_lat, target_lat)
    lon_idx = _nearest_indices(mod_lon, target_lon)
    out = {}
    for name, arr in mod_vars.items():
        out[name] = np.asarray(arr, dtype=float)[np.ix_(lat_idx, lon_idx)].astype(np.float32)
    return out


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


def _find_time_index(ds: xr.Dataset, target_date: dt.date) -> int:
    time_values = pd.to_datetime(ds['valid_time'].values)
    matches = np.where(time_values.date == target_date)[0]
    if len(matches) == 0:
        raise ValueError(f'Cannot find date {target_date} in valid_time.')
    return int(matches[0])


def _assign_era5_from_dataset(
    feature_grids: dict[str, np.ndarray],
    ds: xr.Dataset,
    time_idx: int,
    target_lat: np.ndarray,
    target_lon: np.ndarray,
    lon_cols: np.ndarray,
    source_tag: str,
) -> None:
    lat_values = np.asarray(ds['latitude'].values, dtype=float)
    lon_values = np.asarray(ds['longitude'].values, dtype=float)
    lat_idx = _nearest_indices(lat_values, target_lat)
    lon_for_ds = _to_dataset_lon_values(lon_values, target_lon[lon_cols])
    lon_idx = _nearest_indices(lon_values, lon_for_ds)

    for var_name, da in ds.data_vars.items():
        dims = set(da.dims)
        if 'valid_time' not in dims or 'latitude' not in dims or 'longitude' not in dims:
            continue
        if 'pressure_level' in dims:
            arr = np.asarray(
                da.isel(valid_time=time_idx)
                .transpose('pressure_level', 'latitude', 'longitude')
                .values
            )
            values = arr[:, lat_idx[:, None], lon_idx[None, :]]
            levels = np.asarray(ds['pressure_level'].values)
            for k, level in enumerate(levels):
                col_name = f'{var_name}_{int(round(float(level)))}'
                feature_grids.setdefault(col_name, np.full((target_lat.size, target_lon.size), np.nan, dtype=np.float32))
                feature_grids[col_name][:, lon_cols] = values[k].astype(np.float32)
        else:
            arr = np.asarray(da.isel(valid_time=time_idx).transpose('latitude', 'longitude').values)
            col_name = var_name if var_name not in feature_grids else f'{var_name}_{source_tag}'
            values = arr[lat_idx[:, None], lon_idx[None, :]]
            feature_grids.setdefault(col_name, np.full((target_lat.size, target_lon.size), np.nan, dtype=np.float32))
            feature_grids[col_name][:, lon_cols] = values.astype(np.float32)


def extract_era5_grid(target_date: dt.date, target_lat: np.ndarray, target_lon: np.ndarray) -> dict[str, np.ndarray]:
    feature_grids: dict[str, np.ndarray] = {}
    lon_hours = np.array([lon_to_utc_hour(float(lon), target_lst_hour=TARGET_LST_HOUR) for lon in target_lon], dtype=int)
    ds_cache: dict[str, xr.Dataset] = {}
    try:
        for utc_hour in sorted(set(lon_hours.tolist())):
            lon_cols = np.where(lon_hours == utc_hour)[0]
            paths = _build_era5_paths(target_date, int(utc_hour))
            for tag, path in paths.items():
                key = str(path)
                if key not in ds_cache:
                    ds_cache[key] = xr.open_dataset(path)
                ds = ds_cache[key]
                time_idx = _find_time_index(ds, target_date)
                _assign_era5_from_dataset(feature_grids, ds, time_idx, target_lat, target_lon, lon_cols, tag)
    finally:
        for ds in ds_cache.values():
            ds.close()
    return feature_grids


def build_lnnd(cot: np.ndarray, cer: np.ndarray) -> np.ndarray:
    nd = GAMM * np.power(cot, 0.5) * np.power(cer * 1e-6, -2.5) * 1e-6
    return np.log(nd + 1e-9).astype(np.float32)


def write_daily_grid(target_date: dt.date) -> Path:
    output_path = OUT_DIR / f'{target_date:%Y}' / f'ml_grid_{target_date:%Y%m%d}{LST_TAG}.nc'
    output_path.parent.mkdir(parents=True, exist_ok=True)

    # 如果输出文件已经存在且非空，则跳过
    if output_path.exists() and output_path.stat().st_size > 0:
        print(f'File already exists, skip: {output_path}')
        return output_path

    # 如果文件存在但大小为0，说明上次可能中断，删除后重新生成
    if output_path.exists() and output_path.stat().st_size == 0:
        print(f'Empty file found, remove and regenerate: {output_path}')
        output_path.unlink()

    accu_path = find_accu_sox_file(target_date)

    with xr.open_dataset(accu_path) as accu_ds:
        target_lat = accu_ds['lat'].values.astype(float)
        target_lon = accu_ds['lon'].values.astype(float)
        data_vars = {
            'accu_sox': (('lat', 'lon'), accu_ds['accu_sox'].values.astype(np.float32)),
            'source_sox': (('lat', 'lon'), accu_ds['source_sox'].values.astype(np.float32)),
            'valid_accu_sox': (('lat', 'lon'), accu_ds['valid_accu_sox'].values.astype(np.int8)),
        }

    mod_file = find_mod08_file_for_date(target_date, MOD08_DIR)
    mod_lat, mod_lon, mod_vars = load_mod08_data(mod_file)
    for name, grid in regrid_modis_to_target(mod_lat, mod_lon, mod_vars, target_lat, target_lon).items():
        data_vars[name] = (('lat', 'lon'), grid)

    if 'cot_mod08' in data_vars and 'cer_mod08' in data_vars:
        data_vars['lnnd'] = (('lat', 'lon'), build_lnnd(data_vars['cot_mod08'][1], data_vars['cer_mod08'][1]))

    for name, grid in extract_era5_grid(target_date, target_lat, target_lon).items():
        data_vars[name] = (('lat', 'lon'), grid)

    ds = xr.Dataset(
        data_vars=data_vars,
        coords={'lat': target_lat.astype(np.float32), 'lon': target_lon.astype(np.float32)},
        attrs={
            'date': target_date.isoformat(),
            'year': target_date.year,
            'month': target_date.month,
            'satellite_name': SATELLITE_NAME,
            'target_lst_hour': TARGET_LST_HOUR,
            'mod_file': str(mod_file),
            'accu_sox_file': str(accu_path),
            'description': 'Daily gridded ML data with monthly accu_sox, MODIS cloud variables, and ERA5 variables.',
        },
    )
    ds.to_netcdf(output_path)
    print(f'Saved daily grid: {output_path}')
    return output_path


def main() -> None:
    for target_date in parse_dates_from_argv(sys.argv):
        write_daily_grid(target_date)


if __name__ == '__main__':
    main()
