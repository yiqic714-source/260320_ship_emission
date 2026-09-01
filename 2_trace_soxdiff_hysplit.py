from __future__ import annotations

import calendar
import csv
import datetime as dt
import math
import subprocess
import sys
from collections import defaultdict
from pathlib import Path

import numpy as np
import pandas as pd
import xarray as xr
from util import nearest_utc_for_satellite_overpass_time


SOX_NPZ_PATH = Path('/home/chenyiqi/260320_ship_emission/processed_data/sox_monthly_2000_2022.npz')
CONTROL_PATH = Path('/home/chenyiqi/hysplit/hysplit.v5.4.2_RHEL8.10_public/test_advect_chanel/CONTROL')
RUN_DIR = CONTROL_PATH.parent
HYSPLIT_EXEC = RUN_DIR.parent / 'exec' / 'hyts_std'

SATELLITE_NAME = 'Aqua'  # 'Aqua' or 'Terra'
YEAR_START = 2011
YEAR_END = 2018
RUN_HOURS = -17
CONTROL_HEIGHT = 1000.0
PARTICLE_INDEX_START = 1
PARTICLE_INDEX_END: int | str | None = 'all'
LAT_ABS_MAX = 60.0

OUT_DIR = Path('/home/chenyiqi/260320_ship_emission/processed_data/accu_sox_grid')
METEO_BASE_DIR = '/home/chenyiqi/hysplit/noaa_arl_gdas1deg/'
WEIGHT_SOURCE_CSV = Path('/home/chenyiqi/260320_ship_emission/M22_Fig2_source_data.csv')
WEIGHT_SOURCE_LINES = [35, 43, 51, 59]

MONTH_ABBR = {
    1: 'jan',
    2: 'feb',
    3: 'mar',
    4: 'apr',
    5: 'may',
    6: 'jun',
    7: 'jul',
    8: 'aug',
    9: 'sep',
    10: 'oct',
    11: 'nov',
    12: 'dec',
}


def _satellite_config(satellite_name: str) -> tuple[float, str]:
    name = satellite_name.strip().lower()
    if name == 'aqua':
        return 13.5, '1330'
    if name == 'terra':
        return 10.5, '1030'
    raise ValueError(f'Unsupported SATELLITE_NAME: {satellite_name}')


TARGET_LST_HOUR, LST_TAG = _satellite_config(SATELLITE_NAME)


def normalize_lon(lon: float | np.ndarray) -> float | np.ndarray:
    return ((lon + 180.0) % 360.0) - 180.0


def iter_months(year_start: int, year_end: int) -> list[tuple[int, int]]:
    return [(year, month) for year in range(year_start, year_end + 1) for month in range(1, 13)]


def representative_month_date(year: int, month: int) -> dt.date:
    return dt.date(year, month, min(15, calendar.monthrange(year, month)[1]))


def build_24_utc_slots_for_lst_date(lst_date: dt.date) -> list[dt.datetime]:
    slots = {
        nearest_utc_for_satellite_overpass_time(
            lst_date,
            float(lon),
            target_lst_hour=TARGET_LST_HOUR,
        )
        for lon in np.arange(-179.5, 180.0, 1.0)
    }
    return sorted(slots)


def group_points_by_utc_datetime(
    points: list[tuple[float, float, int, int]],
    lst_date: dt.date,
) -> dict[dt.datetime, list[tuple[float, float, int, int]]]:
    grouped: dict[dt.datetime, list[tuple[float, float, int, int]]] = defaultdict(list)
    for lat, lon, i, j in points:
        utc_dt = nearest_utc_for_satellite_overpass_time(
            lst_date,
            lon,
            target_lst_hour=TARGET_LST_HOUR,
        )
        grouped[utc_dt].append((lat, lon, i, j))
    return dict(grouped)


def meteo_week_index_from_date(date_value: dt.date) -> int:
    return ((date_value.day - 1) // 7) + 1


def build_meteo_path(year: int, month: int, week_index: int) -> str:
    return f'{year}/gdas1.{MONTH_ABBR[month]}{year % 100:02d}.w{week_index}'


def max_week_index_in_month(year: int, month: int) -> int:
    return 5 if calendar.monthrange(year, month)[1] >= 29 else 4


def previous_month(year: int, month: int) -> tuple[int, int]:
    if month == 1:
        return year - 1, 12
    return year, month - 1


def next_month(year: int, month: int) -> tuple[int, int]:
    if month == 12:
        return year + 1, 1
    return year, month + 1


def previous_week(year: int, month: int, week_index: int) -> tuple[int, int, int]:
    if week_index > 1:
        return year, month, week_index - 1
    prev_year, prev_month_value = previous_month(year, month)
    return prev_year, prev_month_value, max_week_index_in_month(prev_year, prev_month_value)


def next_week(year: int, month: int, week_index: int) -> tuple[int, int, int]:
    max_week = max_week_index_in_month(year, month)
    if week_index < max_week:
        return year, month, week_index + 1
    next_year, next_month_value = next_month(year, month)
    return next_year, next_month_value, 1


def find_three_context_meteo_paths(date_value: dt.date) -> list[str]:
    base_dir = Path(METEO_BASE_DIR)
    cur_year = date_value.year
    cur_month = date_value.month
    cur_week = meteo_week_index_from_date(date_value)
    prev_year, prev_month_value, prev_week = previous_week(cur_year, cur_month, cur_week)
    next_year, next_month_value, next_week_index = next_week(cur_year, cur_month, cur_week)

    paths = []
    for year, month, week_index in [
        (prev_year, prev_month_value, prev_week),
        (cur_year, cur_month, cur_week),
        (next_year, next_month_value, next_week_index),
    ]:
        rel_path = build_meteo_path(year, month, week_index)
        if not (base_dir / rel_path).exists():
            raise FileNotFoundError(f'Meteorology file not found: {base_dir / rel_path}')
        paths.append(rel_path)
    return paths


def _find_grid_key(data: np.lib.npyio.NpzFile) -> str:
    for key in ['monthly_sox_grids', 'sox_monthly_grids', 'monthly_grids', 'sox_grids']:
        if key in data.files:
            return key
    if 'annual_mean_grids' in data.files:
        raise KeyError(
            f'{SOX_NPZ_PATH} contains annual_mean_grids, but step 2 needs monthly SOx grids. '
            'Please rerun step 1 so it saves one grid per year-month.'
        )
    raise KeyError(f'Cannot find monthly SOx grid key in {SOX_NPZ_PATH}. Keys: {data.files}')


def load_month_sox_from_npz(npz_path: Path, year: int, month: int) -> tuple[np.ndarray, np.ndarray, np.ndarray]:
    with np.load(npz_path) as data:
        grid_key = _find_grid_key(data)
        grids = data[grid_key].astype(float)
        lat_edges = data['lat_edges'].astype(float)
        lon_edges = data['lon_edges'].astype(float)

        if (
            'years' in data.files
            and 'months' in data.files
            and grids.ndim == 3
            and grids.shape[0] == data['years'].size == data['months'].size
        ):
            matches = np.where((data['years'].astype(int) == year) & (data['months'].astype(int) == month))[0]
            if matches.size == 0:
                raise ValueError(f'No monthly SOx grid for {year}-{month:02d} in {npz_path}')
            return grids[int(matches[0])], lat_edges, lon_edges

        if 'years' in data.files and grids.ndim == 4:
            years = data['years'].astype(int)
            year_matches = np.where(years == year)[0]
            if year_matches.size == 0:
                raise ValueError(f'No year {year} in {npz_path}')
            return grids[int(year_matches[0]), month - 1], lat_edges, lon_edges

    raise ValueError(f'Unsupported monthly SOx NPZ layout in {npz_path}')


def points_from_grid(
    sox_grid: np.ndarray,
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
) -> list[tuple[float, float, int, int]]:
    lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    valid_i, valid_j = np.where(np.isfinite(sox_grid))
    points = []
    for i, j in zip(valid_i, valid_j):
        lat = float(lat_centers[i])
        lon = float(lon_centers[j])
        if abs(lat) < LAT_ABS_MAX:
            points.append((lat, lon, int(i), int(j)))
    if not points:
        raise ValueError('No finite SOx launch points remain after latitude filtering.')
    return points


def select_particle_range(
    points: list[tuple[float, float, int, int]],
    start_1based: int,
    end_1based: int | str | None,
) -> tuple[list[tuple[float, float, int, int]], int, int]:
    if start_1based < 1:
        raise ValueError('PARTICLE_INDEX_START must be >= 1.')
    use_all_end = end_1based is None or (isinstance(end_1based, str) and end_1based.lower() == 'all')
    end_idx = len(points) if use_all_end else min(int(end_1based), len(points))
    selected = points[start_1based - 1:end_idx]
    if not selected:
        raise ValueError('No particles selected.')
    return selected, start_1based, end_idx


def lookup_grid_value(grid: np.ndarray, lat_edges: np.ndarray, lon_edges: np.ndarray, lat: float, lon: float) -> float:
    lon_norm = float(normalize_lon(lon))
    lat_bin = min(max(int(math.floor(lat)), int(lat_edges[0])), int(lat_edges[-2]))
    lon_bin = min(max(int(math.floor(lon_norm)), int(lon_edges[0])), int(lon_edges[-2]))
    return float(grid[lat_bin - int(lat_edges[0]), lon_bin - int(lon_edges[0])])


def build_step_weights_and_std(step_count: int) -> tuple[np.ndarray, np.ndarray]:
    all_lines = WEIGHT_SOURCE_CSV.read_text(encoding='utf-8').splitlines()
    rows = []
    for line_no in WEIGHT_SOURCE_LINES:
        if not 1 <= line_no <= len(all_lines):
            continue
        parts = list(csv.reader([all_lines[line_no - 1].strip()]))[0]
        if not parts or 'anomaly' not in parts[0].lower():
            continue
        vals = []
        for token in parts[1:]:
            try:
                vals.append(float(token) * 100.0 - 100.0)
            except ValueError:
                pass
        if vals:
            rows.append(np.asarray(vals, dtype=float))
    if not rows:
        raise ValueError(f'No valid weight rows found in {WEIGHT_SOURCE_CSV}')
    use_len = min(step_count, min(row.size for row in rows))
    norm_rows = []
    for row in rows:
        cut = row[:use_len]
        total = float(np.sum(cut))
        norm_rows.append(cut / total if abs(total) > 1e-12 else np.full(use_len, 1.0 / use_len))
    stack = np.vstack(norm_rows)
    return np.mean(stack, axis=0), np.std(stack, axis=0)


def read_step_weights_csv(input_csv: Path) -> np.ndarray:
    df = pd.read_csv(input_csv)
    weights = pd.to_numeric(df['step_weight_ave'], errors='coerce').to_numpy(dtype=float)
    weights = weights[np.isfinite(weights)]
    if weights.size == 0:
        raise ValueError(f'No finite step weights in {input_csv}')
    return weights


def write_step_weights_csv(step_weights: np.ndarray, step_stds: np.ndarray, output_csv: Path) -> None:
    output_csv.parent.mkdir(parents=True, exist_ok=True)
    with output_csv.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(['step_index_0based', 'step_weight_ave', 'step_weight_std'])
        for idx, weight in enumerate(step_weights):
            writer.writerow([idx, float(weight), float(step_stds[idx])])


def compute_weighted_mean(values: list[float], weights: np.ndarray) -> float:
    arr = np.asarray(values, dtype=float)
    use_len = min(arr.size, weights.size)
    if use_len <= 0:
        return float('nan')
    arr = arr[:use_len]
    weights = weights[:use_len]
    mask = np.isfinite(arr) & np.isfinite(weights)
    if not np.any(mask):
        return float('nan')
    return float(np.average(arr[mask], weights=weights[mask]))


def update_control(
    control_path: Path,
    points: list[tuple[float, float, int, int]],
    meteo_paths: list[str],
    output_name: str,
    target_dt: dt.datetime,
    run_hours: int,
) -> None:
    lines = control_path.read_text().splitlines()
    particle_start = 2

    def find_footer_start_idx() -> int:
        for idx in range(len(lines) - 8, particle_start - 1, -1):
            try:
                nmet = int(lines[idx + 3].strip())
                out_name_idx = idx + 4 + 2 * nmet + 1
            except (ValueError, IndexError):
                continue
            if nmet > 0 and out_name_idx < len(lines):
                return idx
        raise ValueError(f'Cannot locate valid CONTROL footer in {control_path}')

    footer_lines = lines[find_footer_start_idx():]
    particle_lines = [f'{lat:.1f} {lon:.1f} {CONTROL_HEIGHT:.1f}' for lat, lon, _, _ in points]
    lines[0] = f'{target_dt.year % 100:02d} {target_dt.month:02d} {target_dt.day:02d} {target_dt.hour:02d} {target_dt.minute:02d}'
    lines[1] = str(len(points))
    updated_lines = lines[:particle_start] + particle_lines + footer_lines
    footer_offset = particle_start + len(particle_lines)
    updated_lines[footer_offset] = str(run_hours)
    nmet_old = int(updated_lines[footer_offset + 3].strip())
    meteo_start = footer_offset + 4
    meteo_end = meteo_start + 2 * nmet_old
    rest_after_meteo = updated_lines[meteo_end:]
    meteo_lines = [line for meteo_path in meteo_paths for line in (METEO_BASE_DIR, meteo_path)]
    updated_lines[footer_offset + 3] = str(len(meteo_paths))
    updated_lines = updated_lines[:meteo_start] + meteo_lines + rest_after_meteo
    output_name_idx = meteo_start + len(meteo_lines) + 1
    updated_lines[output_name_idx] = output_name
    control_path.write_text('\n'.join(updated_lines) + '\n')


def run_hyts(run_dir: Path, executable: Path) -> None:
    if not executable.exists():
        raise FileNotFoundError(f'HYSPLIT executable not found: {executable}')
    log_path = run_dir / 'rsl_out.hysplit'
    with log_path.open('w', encoding='utf-8') as log_file:
        subprocess.run([str(executable)], cwd=run_dir, check=True, stdout=log_file, stderr=log_file)


def extract_all_particle_latlon_from_output(output_file: Path) -> dict[int, list[tuple[float, float]]]:
    rows: dict[int, list[tuple[float, float]]] = {}
    lines = output_file.read_text(encoding='utf-8').splitlines()
    data_start = next((idx + 1 for idx, line in enumerate(lines) if 'PRESSURE' in line), None)
    if data_start is None:
        raise ValueError(f"Cannot find data section marker 'PRESSURE' in {output_file}")
    for line in lines[data_start:]:
        parts = line.split()
        if len(parts) < 12:
            continue
        rows.setdefault(int(parts[0]), []).append((float(parts[9]), float(normalize_lon(float(parts[10])))))
    if not rows:
        raise ValueError(f'No trajectory rows found in {output_file}')
    return rows


def save_accu_sox_grid(
    output_path: Path,
    accu_sox: np.ndarray,
    source_sox: np.ndarray,
    lat_edges: np.ndarray,
    lon_edges: np.ndarray,
    year: int,
    month: int,
) -> None:
    output_path.parent.mkdir(parents=True, exist_ok=True)
    lat = 0.5 * (lat_edges[:-1] + lat_edges[1:])
    lon = 0.5 * (lon_edges[:-1] + lon_edges[1:])
    ds = xr.Dataset(
        data_vars={
            'accu_sox': (('lat', 'lon'), accu_sox.astype(np.float32)),
            'source_sox': (('lat', 'lon'), source_sox.astype(np.float32)),
            'valid_accu_sox': (('lat', 'lon'), np.isfinite(accu_sox).astype(np.int8)),
        },
        coords={
            'lat': lat.astype(np.float32),
            'lon': lon.astype(np.float32),
            'lat_edges': lat_edges.astype(np.float32),
            'lon_edges': lon_edges.astype(np.float32),
        },
        attrs={
            'year': year,
            'month': month,
            'satellite_name': SATELLITE_NAME,
            'target_lst_hour': TARGET_LST_HOUR,
            'run_hours': RUN_HOURS,
            'control_height_m': CONTROL_HEIGHT,
            'description': 'Monthly gridded accumulated SOx along HYSPLIT trajectories.',
        },
    )
    ds.to_netcdf(output_path)


def run_one_month(year: int, month: int) -> Path:
    sox_grid, lat_edges, lon_edges = load_month_sox_from_npz(SOX_NPZ_PATH, year, month)
    all_points = points_from_grid(sox_grid, lat_edges, lon_edges)
    run_points, selected_start, selected_end = select_particle_range(
        all_points,
        PARTICLE_INDEX_START,
        PARTICLE_INDEX_END,
    )
    month_date = representative_month_date(year, month)
    points_by_utc = group_points_by_utc_datetime(run_points, month_date)
    utc_slots = build_24_utc_slots_for_lst_date(month_date)
    output_path = OUT_DIR / f'{year}' / f'accu_sox_{year}{month:02d}_{LST_TAG}.nc'
    step_weights_csv_path = OUT_DIR / 'step_weights.csv'
    step_weights = read_step_weights_csv(step_weights_csv_path) if step_weights_csv_path.exists() else None
    accu_sox = np.full_like(sox_grid, np.nan, dtype=np.float32)

    print(f'{year}-{month:02d}: finite launch points={len(all_points)}, selected={selected_start}..{selected_end}')
    for utc_dt in utc_slots:
        hour_points = points_by_utc.get(utc_dt, [])
        if not hour_points:
            continue
        meteo_paths = find_three_context_meteo_paths(utc_dt.date())
        output_name = 'output'
        update_control(CONTROL_PATH, hour_points, meteo_paths, output_name, utc_dt, RUN_HOURS)
        print(f'  HYSPLIT UTC={utc_dt:%Y-%m-%d %H:%M}, particles={len(hour_points)}')
        run_hyts(RUN_DIR, HYSPLIT_EXEC)
        tracks_by_particle = extract_all_particle_latlon_from_output(RUN_DIR / output_name)

        for particle_idx, track in sorted(tracks_by_particle.items()):
            if particle_idx < 1 or particle_idx > len(hour_points):
                continue
            _, _, i, j = hour_points[particle_idx - 1]
            step_values = [lookup_grid_value(sox_grid, lat_edges, lon_edges, lat, lon) for lat, lon in track]
            if step_weights is None:
                step_weights, step_stds = build_step_weights_and_std(len(step_values))
                write_step_weights_csv(step_weights, step_stds, step_weights_csv_path)
            accu_sox[i, j] = compute_weighted_mean(step_values, step_weights)

    save_accu_sox_grid(output_path, accu_sox, sox_grid, lat_edges, lon_edges, year, month)
    print(f'Saved monthly accu_sox grid: {output_path}')
    return output_path


def parse_months_from_argv(argv: list[str]) -> list[tuple[int, int]]:
    if len(argv) == 1:
        return iter_months(YEAR_START, YEAR_END)
    if len(argv) == 3:
        return [(int(argv[1]), int(argv[2]))]
    raise SystemExit('Usage: python 2_trace_soxdiff_hysplit.py [yyyy mm]')


def main() -> None:
    for year, month in parse_months_from_argv(sys.argv):
        run_one_month(year, month)


if __name__ == '__main__':
    main()
