from __future__ import annotations

import csv
import datetime as dt
import calendar
from collections import defaultdict
import math
import subprocess
import sys
from pathlib import Path

import cartopy.crs as ccrs
import cartopy.feature as cfeature
import cartopy.io.shapereader as shpreader
import matplotlib.pyplot as plt
from matplotlib.colors import LogNorm
import numpy as np
import pandas as pd
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep
from util import nearest_utc_for_satellite_overpass_time


NPZ_PATH = Path('/home/chenyiqi/260320_ship_emission/processed_data/soxdiff_monthly_2019m2020.npz')
CONTROL_PATH = Path('/home/chenyiqi/hysplit/hysplit.v5.4.2_RHEL8.10_public/test_advect_chanel/CONTROL')
RUN_DIR = CONTROL_PATH.parent
HYSPLIT_EXEC = RUN_DIR.parent / 'exec' / 'hyts_std'

SATELLITE_NAME = 'Aqua'  # 'Aqua' or 'Terra'


def _satellite_config(satellite_name: str) -> tuple[float, str]:
	name = satellite_name.strip().lower()
	if name == 'aqua':
		return 13.5, '1330'
	if name == 'terra':
		return 10.5, '1030'
	raise ValueError(f'Unsupported SATELLITE_NAME: {satellite_name}')


TARGET_LST_HOUR, LST_TAG = _satellite_config(SATELLITE_NAME)
RUN_HOURS = -20
CONTROL_HEIGHT = 1000.0
PARTICLE_INDEX_START = 1
PARTICLE_INDEX_END = 'all'
WEIGHT_MODE = 'uniform'
RESULT_CSV_DIR = Path('/home/chenyiqi/260320_ship_emission/processed_data/hysplit_rsl')
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


def normalize_lon(lon: float) -> float:
	return ((lon + 180.0) % 360.0) - 180.0


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
	points: list[tuple[float, float]],
	lst_date: dt.date,
) -> dict[dt.datetime, list[tuple[float, float]]]:
	grouped: dict[dt.datetime, list[tuple[float, float]]] = defaultdict(list)
	for lat, lon in points:
		utc_dt = nearest_utc_for_satellite_overpass_time(
			lst_date,
			lon,
			target_lst_hour=TARGET_LST_HOUR,
		)
		grouped[utc_dt].append((lat, lon))
	return dict(grouped)


def meteo_week_index_from_date(date_value: dt.date) -> int:
	return ((date_value.day - 1) // 7) + 1


def build_meteo_path(year: int, month: int, week_index: int) -> str:
	month_abbr = MONTH_ABBR[month]
	return f'{year}/gdas1.{month_abbr}{year % 100:02d}.w{week_index}'


def max_week_index_in_month(year: int, month: int) -> int:
	# w5 exists if the month has day 29 or later.
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

	week_triplet = [
		(prev_year, prev_month_value, prev_week),
		(cur_year, cur_month, cur_week),
		(next_year, next_month_value, next_week_index),
	]

	paths: list[str] = []
	for year, month, week_index in week_triplet:
		rel_path = build_meteo_path(year, month, week_index)
		if not (base_dir / rel_path).exists():
			raise FileNotFoundError(
				f'Meteorology file not found for {year}-{month:02d} w{week_index}: '
				f'{base_dir / rel_path}'
			)
		paths.append(rel_path)

	return paths


def load_month_points_from_npz(
	npz_path: Path,
	month: int,
) -> tuple[list[tuple[float, float]], np.ndarray, np.ndarray, np.ndarray]:
	with np.load(npz_path) as data:
		months = data['months'].astype(int)
		month_matches = np.where(months == month)[0]
		if month_matches.size == 0:
			raise ValueError(f'Month {month} not found in {npz_path}.')
		month_idx = int(month_matches[0])
		lat_edges = data['lat_edges'].astype(float)
		lon_edges = data['lon_edges'].astype(float)
		diff_grid = data['diff_grids'][month_idx].astype(float)

	lat_centers = 0.5 * (lat_edges[:-1] + lat_edges[1:])
	lon_centers = 0.5 * (lon_edges[:-1] + lon_edges[1:])
	valid_i, valid_j = np.where(np.isfinite(diff_grid))
	points = [(float(lat_centers[i]), float(lon_centers[j])) for i, j in zip(valid_i, valid_j)]
	if not points:
		raise ValueError(f'No finite SOx difference points found for month={month}.')
	return points, diff_grid, lat_edges, lon_edges


def select_particle_range(
	points: list[tuple[float, float]],
	start_1based: int,
	end_1based: int | str | None,
) -> tuple[list[tuple[float, float]], int, int]:
	if start_1based < 1:
		raise ValueError('PARTICLE_INDEX_START must be >= 1.')

	use_all_end = end_1based is None or (isinstance(end_1based, str) and end_1based.lower() == 'all')
	if use_all_end:
		end_idx = len(points)
	else:
		if not isinstance(end_1based, int) or end_1based < start_1based:
			raise ValueError('PARTICLE_INDEX_END must be >= PARTICLE_INDEX_START, or None/"all".')
		end_idx = min(end_1based, len(points))

	if start_1based > len(points):
		raise ValueError(f'PARTICLE_INDEX_START={start_1based} exceeds available particles={len(points)}.')

	selected = points[start_1based - 1:end_idx]
	if not selected:
		raise ValueError('No particles selected by PARTICLE_INDEX_START/PARTICLE_INDEX_END.')
	return selected, start_1based, end_idx


def build_land_prepared_geometry():
	land_path = shpreader.natural_earth(
		resolution='110m',
		category='physical',
		name='land',
	)
	reader = shpreader.Reader(land_path)
	land_geom = unary_union(list(reader.geometries()))
	return prep(land_geom)


def track_reaches_land(track_pairs: list[tuple[float, float]], land_prepared) -> bool:
	for lat, lon in track_pairs:
		if land_prepared.covers(Point(float(normalize_lon(lon)), float(lat))):
			return True
	return False


def lookup_sox_diff(
	diff_grid: np.ndarray,
	lat_edges: np.ndarray,
	lon_edges: np.ndarray,
	lat: float,
	lon: float,
) -> float:
	lon_norm = normalize_lon(lon)
	lat_bin = int(math.floor(lat))
	lon_bin = int(math.floor(lon_norm))
	lat_bin = min(max(lat_bin, int(lat_edges[0])), int(lat_edges[-2]))
	lon_bin = min(max(lon_bin, int(lon_edges[0])), int(lon_edges[-2]))
	lat_idx = lat_bin - int(lat_edges[0])
	lon_idx = lon_bin - int(lon_edges[0])
	return float(diff_grid[lat_idx, lon_idx])


def build_step_weights_and_std(step_count: int, mode: str) -> tuple[np.ndarray, np.ndarray]:
	if step_count <= 0:
		raise ValueError('step_count must be positive.')
	if mode != 'uniform':
		raise ValueError(f'Unsupported WEIGHT_MODE: {mode}')

	all_lines = WEIGHT_SOURCE_CSV.read_text(encoding='utf-8').splitlines()
	line_arrays: list[np.ndarray] = []
	for line_no in WEIGHT_SOURCE_LINES:
		if line_no < 1 or line_no > len(all_lines):
			continue
		raw_line = all_lines[line_no - 1].strip()
		if not raw_line:
			continue
		parts = list(csv.reader([raw_line]))[0]
		if not parts or 'anomaly' not in parts[0].strip().lower():
			continue
		vals: list[float] = []
		for token in parts[1:]:
			t = token.strip()
			if t == '':
				continue
			try:
				vals.append(float(t) * 100 - 100)
			except ValueError:
				continue
		if vals:
			line_arrays.append(np.asarray(vals, dtype=float))

	if not line_arrays:
		raise ValueError(
			f'No valid weight rows found in {WEIGHT_SOURCE_CSV} for lines {WEIGHT_SOURCE_LINES}. '
			'Please verify line numbers.'
		)

	use_len = min(step_count, min(arr.size for arr in line_arrays))
	if use_len <= 0:
		raise ValueError('Weight rows are empty after parsing.')

	normalized_rows: list[np.ndarray] = []
	for arr in line_arrays:
		arr_cut = arr[:use_len]
		total = float(np.sum(arr_cut))
		if np.isfinite(total) and abs(total) > 1e-12:
			normalized_rows.append(arr_cut / total)
		else:
			normalized_rows.append(np.full(use_len, 1.0 / use_len, dtype=float))

	stacked = np.vstack(normalized_rows)
	return np.mean(stacked, axis=0), np.std(stacked, axis=0)


def read_step_weights_csv(input_csv: Path) -> np.ndarray:
	if not input_csv.exists():
		raise FileNotFoundError(f'Cannot find step weights CSV: {input_csv}')
	df = pd.read_csv(input_csv)
	if 'step_weight_ave' not in df.columns:
		raise ValueError(f'step_weight_ave column missing in {input_csv}')
	weights = pd.to_numeric(df['step_weight_ave'], errors='coerce').to_numpy(dtype=float)
	weights = weights[np.isfinite(weights)]
	if weights.size == 0:
		raise ValueError(f'No finite step_weight_ave values in {input_csv}')
	return weights


def compute_weighted_mean(values: list[float], weights: np.ndarray) -> float:
	arr = np.asarray(values, dtype=float)
	use_len = min(arr.size, int(weights.size))
	if use_len <= 0:
		return float('nan')
	arr = arr[:use_len]
	weights = weights[:use_len]
	if not np.all(np.isfinite(arr)):
		return float('nan')
	return float(np.average(arr, weights=weights))


def parse_target_date_from_argv(argv: list[str]) -> dt.date:
	if len(argv) != 4:
		raise SystemExit('Usage: python 2_advect_chanel.py yyyy mm dd')

	try:
		year = int(argv[1])
		month = int(argv[2])
		day = int(argv[3])
		return dt.date(year, month, day)
	except ValueError as exc:
		raise SystemExit(f'Invalid date arguments: {exc}') from exc


def update_control(
	control_path: Path,
	points: list[tuple[float, float]],
	meteo_paths: list[str],
	output_name: str,
	target_dt: dt.datetime,
	run_hours: int,
) -> None:
	lines = control_path.read_text().splitlines()
	if len(lines) < 6:
		raise ValueError(f'Unexpected CONTROL format in {control_path}.')

	particle_start = 2

	def _find_footer_start_idx() -> int:
		for idx in range(len(lines) - 8, particle_start - 1, -1):
			try:
				int(lines[idx].strip())
				int(lines[idx + 1].strip())
				float(lines[idx + 2].strip())
				nmet = int(lines[idx + 3].strip())
			except (ValueError, IndexError):
				continue
			if nmet <= 0:
				continue
			out_dir_idx = idx + 4 + 2 * nmet
			out_name_idx = out_dir_idx + 1
			if out_name_idx >= len(lines):
				continue
			if int(lines[idx].strip()) != run_hours:
				continue
			return idx

		for idx in range(len(lines) - 8, particle_start - 1, -1):
			try:
				int(lines[idx].strip())
				int(lines[idx + 1].strip())
				float(lines[idx + 2].strip())
				nmet = int(lines[idx + 3].strip())
			except (ValueError, IndexError):
				continue
			if nmet <= 0:
				continue
			out_dir_idx = idx + 4 + 2 * nmet
			out_name_idx = out_dir_idx + 1
			if out_name_idx < len(lines):
				return idx

		raise ValueError(f'Cannot locate valid CONTROL footer in {control_path}.')

	footer_start = _find_footer_start_idx()
	footer_lines = lines[footer_start:]
	if len(footer_lines) < 8:
		raise ValueError(f'CONTROL footer is unexpectedly short in {control_path}.')

	new_particle_lines = [f'{lat:.1f} {lon:.1f} {CONTROL_HEIGHT:.1f}' for lat, lon in points]
	lines[0] = (
		f'{target_dt.year % 100:02d} {target_dt.month:02d} {target_dt.day:02d} '
		f'{target_dt.hour:02d} {target_dt.minute:02d}'
	)
	lines[1] = str(len(points))
	updated_lines = lines[:particle_start] + new_particle_lines + footer_lines
	new_run_hours_idx = particle_start + len(new_particle_lines)
	updated_lines[new_run_hours_idx] = str(run_hours)

	footer_offset = new_run_hours_idx
	nmet_old = int(updated_lines[footer_offset + 3].strip())
	meteo_start = footer_offset + 4
	meteo_end = meteo_start + 2 * nmet_old
	rest_after_meteo = updated_lines[meteo_end:]

	meteo_lines: list[str] = []
	for meteo_path in meteo_paths:
		meteo_lines.extend([METEO_BASE_DIR, meteo_path])

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
	particle_rows: dict[int, list[tuple[float, float]]] = {}
	with output_file.open('r', encoding='utf-8') as handle:
		lines = handle.readlines()

	data_start = None
	for idx, line in enumerate(lines):
		if 'PRESSURE' in line:
			data_start = idx + 1
			break

	if data_start is None:
		raise ValueError(f"Cannot find data section marker 'PRESSURE' in {output_file}")

	for line in lines[data_start:]:
		parts = line.split()
		if len(parts) < 12:
			continue
		particle_idx = int(parts[0])
		lat = float(parts[9])
		lon = normalize_lon(float(parts[10]))
		particle_rows.setdefault(particle_idx, []).append((lat, lon))

	if not particle_rows:
		raise ValueError(f'No trajectory rows found in {output_file}.')
	return particle_rows


def write_step_weights_csv(step_weights: np.ndarray, step_stds: np.ndarray, output_csv: Path) -> None:
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	with output_csv.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.writer(handle)
		writer.writerow(['step_index_0based', 'step_weight_ave', 'step_weight_std'])
		for idx, weight in enumerate(step_weights):
			writer.writerow([idx, float(weight), float(step_stds[idx])])

def write_results_csv(rows: list[dict[str, float | int | str]], output_csv: Path) -> None:
	fieldnames = [
		't0_lat',
		't0_lon',
		'weighted_sox_diff',
	]
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	with output_csv.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)

def plot_kept_t0_positions(
	kept_t0_records: list[tuple[dt.datetime, float, float, float]],
	output_png: Path,
) -> None:
	fig = plt.figure(figsize=(11, 5.5), dpi=220)
	ax = plt.axes(projection=ccrs.PlateCarree())
	ax.set_global()
	ax.coastlines(resolution='110m', linewidth=0.6)
	ax.add_feature(cfeature.LAND, facecolor='lightgray', edgecolor='none', zorder=0)
	ax.gridlines(draw_labels=False, linewidth=0.4, linestyle='--', alpha=0.35)

	if kept_t0_records:
		t0_lons = [normalize_lon(lon) for _, _, lon, _ in kept_t0_records]
		t0_lats = [lat for _, lat, _, _ in kept_t0_records]
		sox_values = np.array([sox for _, _, _, sox in kept_t0_records], dtype=float)
		positive_mask = np.isfinite(sox_values) & (sox_values > 0.0)
		if not np.any(positive_mask):
			print('No positive SOx diff values available for logarithmic color scale.')
			positive_mask = np.isfinite(sox_values)

		plot_lons = np.asarray(t0_lons, dtype=float)[positive_mask]
		plot_lats = np.asarray(t0_lats, dtype=float)[positive_mask]
		plot_vals = sox_values[positive_mask]
		min_positive = float(np.nanmin(plot_vals[plot_vals > 0.0])) if np.any(plot_vals > 0.0) else 1e-12
		norm = LogNorm(vmin=min_positive, vmax=float(np.nanmax(plot_vals))) if np.any(plot_vals > 0.0) else None
		sc = ax.scatter(
			plot_lons,
			plot_lats,
			c=plot_vals,
			s=10,
			cmap='viridis',
			norm=norm,
			alpha=0.85,
			linewidths=0,
			transform=ccrs.PlateCarree(),
			zorder=3,
		)
		cbar = fig.colorbar(sc, ax=ax, orientation='vertical', fraction=0.03, pad=0.02)
		cbar.set_label('Weighted SOx difference')

	ax.set_title('Global Distribution of Kept Particles at T0 UTC (colored by SOx diff)')
	output_png.parent.mkdir(parents=True, exist_ok=True)
	fig.savefig(output_png, bbox_inches='tight')
	print(f'Saved figure: {output_png}')
	plt.close(fig)


def main() -> None:
	target_date = parse_target_date_from_argv(sys.argv)
	run_points_all, month_diff_grid, lat_edges, lon_edges = load_month_points_from_npz(
		NPZ_PATH,
		target_date.month,
	)
	run_points, selected_start, selected_end = select_particle_range(
		run_points_all,
		start_1based=PARTICLE_INDEX_START,
		end_1based=PARTICLE_INDEX_END,
	)
	points_by_utc = group_points_by_utc_datetime(run_points, target_date)
	utc_slots = build_24_utc_slots_for_lst_date(target_date)
	result_csv_path = RESULT_CSV_DIR / f'{target_date:%Y}/soxdiff_track_mean_{target_date:%Y%m%d}{LST_TAG}.csv'
	kept_t0_png = RESULT_CSV_DIR / f'kept_particles_t0_map_{target_date:%Y%m%d}.png'
	step_weights_csv_path = RESULT_CSV_DIR / 'step_weights.csv'

	# print(f'TARGET_LST_HOUR: {TARGET_LST_HOUR}')
	print(f'Finite SOx-diff particles in NPZ (month={target_date.month}): {len(run_points_all)}')
	print(f'Selected particle range (1-based, inclusive): {selected_start}..{selected_end}')
	# print(f'Using run_hours={RUN_HOURS}')

	saved_rows: list[dict[str, float | int | str]] = []
	kept_t0_records: list[tuple[dt.datetime, float, float, float]] = []
	total_particles_simulated = 0
	kept_particles = 0
	runs_executed = 0
	step_weights_saved = False
	step_weights_ave: np.ndarray | None = None
	step_weights_file_exists = step_weights_csv_path.exists()
	if step_weights_file_exists:
		step_weights_saved = True
		step_weights_ave = read_step_weights_csv(step_weights_csv_path)

	for utc_dt in utc_slots:
		hour_points = points_by_utc.get(utc_dt, [])
		if not hour_points:
			continue

		meteo_paths = find_three_context_meteo_paths(utc_dt.date())
		output_name = 'output'

		update_control(CONTROL_PATH, hour_points, meteo_paths, output_name, utc_dt, RUN_HOURS)
		print(f'T0 datetime (UTC): {utc_dt:%Y-%m-%d %H:%M}, particles={len(hour_points)}')

		run_hyts(RUN_DIR, HYSPLIT_EXEC)
		runs_executed += 1

		output_file = RUN_DIR / output_name
		tracks_by_particle = extract_all_particle_latlon_from_output(output_file)
		total_particles_simulated += len(tracks_by_particle)

		for particle_idx in sorted(tracks_by_particle.keys()):
			track = tracks_by_particle[particle_idx]
			step_sox_diff = [
				lookup_sox_diff(month_diff_grid, lat_edges, lon_edges, lat, lon)
				for lat, lon in track
			]
			if not step_weights_saved:
				step_weights_ave, step_stds = build_step_weights_and_std(len(step_sox_diff), WEIGHT_MODE)
				write_step_weights_csv(step_weights_ave, step_stds, step_weights_csv_path)
				print(f'Saved step weights: {step_weights_csv_path}')
				step_weights_saved = True
			if step_weights_ave is None:
				raise ValueError('step_weights_ave is not initialized.')
			weighted_sox_diff = compute_weighted_mean(step_sox_diff, step_weights_ave)
			if not math.isfinite(weighted_sox_diff):
				continue
			t0_lat, t0_lon = track[0]
			kept_particles += 1
			kept_t0_records.append((utc_dt, float(t0_lat), float(t0_lon), float(weighted_sox_diff)))
			saved_rows.append(
				{
					't0_lat': float(t0_lat),
					't0_lon': float(t0_lon),
					'weighted_sox_diff': weighted_sox_diff,
				}
			)
	write_results_csv(saved_rows, result_csv_path)
	# plot_kept_t0_positions(kept_t0_records, kept_t0_png)

	print(f'Saved CSV: {result_csv_path}')
	print(f'Kept particles: {kept_particles}/{total_particles_simulated}')


if __name__ == '__main__':
	main()
