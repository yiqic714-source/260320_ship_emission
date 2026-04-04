from __future__ import annotations

import csv
import datetime as dt
from collections import defaultdict
import math
import subprocess
import sys
from pathlib import Path

import cartopy.io.shapereader as shpreader
from shapely.geometry import Point
from shapely.ops import unary_union
from shapely.prepared import prep


CSV_PATH = Path('/home/chenyiqi/260320_ship_emission/processed_data/sox_selected_points_2019_minus_2020_top50.csv')
CONTROL_PATH = Path('/home/chenyiqi/hysplit/hysplit.v5.4.2_RHEL8.10_public/test_advect_chanel/CONTROL')
RUN_DIR = CONTROL_PATH.parent
HYSPLIT_EXEC = RUN_DIR.parent / 'exec' / 'hyts_std'

TARGET_LST_HOUR = 13.5
RUN_HOURS = -5
NEARBY_DEGREES = 0
# Select particle index range on expanded points (1-based, inclusive).
# Example: 1..100 runs the first 100 particles; 200..'all' runs from #200 to the end.
PARTICLE_INDEX_START = 1
PARTICLE_INDEX_END = 'all'
RESULT_CSV_DIR = Path('/home/chenyiqi/260320_ship_emission/processed_data/hysplit_rsl')
METEO_BASE_DIR = '/home/chenyiqi/hysplit/noaa_arl_gdas1deg/'
# Keep pair matching robust by snapping to 0.5-degree grid (reference is in *.5 form).
PAIR_GRID_STEP = 0.5

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


def load_points(csv_path: Path, month: int, count: int | str | None) -> list[tuple[float, float]]:
	use_all = count is None or (isinstance(count, str) and count.lower() == 'all')
	if not use_all and (not isinstance(count, int) or count <= 0):
		raise ValueError('count must be a positive integer, None, or "all".')

	points: list[tuple[float, float]] = []
	with csv_path.open(newline='') as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			if int(row['month']) != month:
				continue
			points.append((float(row['lat']), float(row['lon'])))

	if not points:
		raise ValueError(f'No points found for month={month}.')

	# Remove duplicates while preserving order.
	seen: set[tuple[float, float]] = set()
	unique_points: list[tuple[float, float]] = []
	for lat, lon in points:
		key = (round(lat, 6), round(lon, 6))
		if key in seen:
			continue
		seen.add(key)
		unique_points.append((lat, lon))

	if use_all:
		return unique_points
	if len(unique_points) < count:
		raise ValueError(f'Only found {len(unique_points)} points for month={month}, need {count}.')
	return unique_points[:count]


def normalize_lon(lon: float) -> float:
	return ((lon + 180.0) % 360.0) - 180.0


def _circular_mean_lon(lons: list[float]) -> float:
	if not lons:
		raise ValueError('Cannot compute mean longitude from empty list.')
	sin_sum = sum(math.sin(math.radians(lon)) for lon in lons)
	cos_sum = sum(math.cos(math.radians(lon)) for lon in lons)
	return math.degrees(math.atan2(sin_sum, cos_sum))


def nearest_utc_hour_for_lst1330(lon: float) -> int:
	utc_float = TARGET_LST_HOUR - (lon / 15.0)
	utc_mod = utc_float % 24.0
	return int(math.floor(utc_mod + 0.5)) % 24


def round_to_nearest_utc_hour_half_down(time_value: dt.datetime) -> dt.datetime:
	base = time_value.replace(minute=0, second=0, microsecond=0)
	delta_seconds = (time_value - base).total_seconds()
	# For exact xx:30:00 ties, round down to avoid creating a 25th UTC slot.
	if delta_seconds > 1800:
		base += dt.timedelta(hours=1)
	return base


def nearest_utc_datetime_for_lst1330(lst_date: dt.date, lon: float) -> dt.datetime:
	lst_dt = dt.datetime(lst_date.year, lst_date.month, lst_date.day, 13, 30)
	utc_dt = lst_dt - dt.timedelta(hours=(lon / 15.0))
	return round_to_nearest_utc_hour_half_down(utc_dt)


def build_24_utc_slots_for_lst_date(lst_date: dt.date) -> list[dt.datetime]:
	# For LST at yyyy-mm-dd 13:30 and lon in [-180, 180), UTC lies in [D 01:30, D+1 01:30),
	# which rounds to these 24 hourly slots: D 02:00 ... D+1 01:00.
	start = dt.datetime(lst_date.year, lst_date.month, lst_date.day, 2, 0)
	return [start + dt.timedelta(hours=i) for i in range(24)]


def group_points_by_utc_datetime(
	points: list[tuple[float, float]],
	lst_date: dt.date,
) -> dict[dt.datetime, list[tuple[float, float]]]:
	grouped: dict[dt.datetime, list[tuple[float, float]]] = defaultdict(list)
	for lat, lon in points:
		utc_dt = nearest_utc_datetime_for_lst1330(lst_date, lon)
		grouped[utc_dt].append((lat, lon))
	return dict(grouped)


def meteo_week_index_from_date(date_value: dt.date) -> int:
	# GDAS weekly file convention: w1=days 1-7, w2=8-14, w3=15-21, w4=22-28, w5=29-end.
	return ((date_value.day - 1) // 7) + 1


def build_meteo_path(year: int, month: int, week_index: int) -> str:
	month_abbr = MONTH_ABBR[month]
	return f'{year}/gdas1.{month_abbr}{year % 100:02d}.w{week_index}'


def build_land_prepared_geometry():
	land_path = shpreader.natural_earth(
		resolution='110m',
		category='physical',
		name='land',
	)
	reader = shpreader.Reader(land_path)
	land_geom = unary_union(list(reader.geometries()))
	return prep(land_geom)


def expand_to_nearby_ocean_points(
	base_points: list[tuple[float, float]],
	distance_deg: int,
	land_prepared,
) -> list[tuple[float, float]]:
	if distance_deg < 0:
		raise ValueError('distance_deg must be >= 0.')

	unique: set[tuple[float, float]] = set()
	expanded: list[tuple[float, float]] = []
	for base_lat, base_lon in base_points:
		for dlat in range(-distance_deg, distance_deg + 1):
			for dlon in range(-distance_deg, distance_deg + 1):
				lat = base_lat + dlat
				if lat < -90.0 or lat > 90.0:
					continue
				lon = normalize_lon(base_lon + dlon)
				is_land = land_prepared.covers(Point(float(lon), float(lat)))
				if is_land:
					continue
				key = (round(lat, 6), round(lon, 6))
				if key in unique:
					continue
				unique.add(key)
				expanded.append((lat, lon))

	if not expanded:
		raise ValueError(f'No ocean points found after {NEARBY_DEGREES}-degree expansion.')
	return expanded


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


def _snap_to_grid(value: float, step: float) -> float:
	if step <= 0:
		raise ValueError('Grid step must be > 0.')
	return round(value / step) * step


def _snap_latlon_pair(lat: float, lon: float, step: float) -> tuple[float, float]:
	lat_s = _snap_to_grid(lat, step)
	lon_s = _snap_to_grid(normalize_lon(lon), step)
	return float(lat_s), float(normalize_lon(lon_s))


def build_pair_set(points: list[tuple[float, float]], grid_step: float) -> set[tuple[float, float]]:
	return {_snap_latlon_pair(lat, lon, grid_step) for lat, lon in points}


def parse_target_date_from_argv(argv: list[str]) -> dt.date:
	if len(argv) != 4:
		raise SystemExit('Usage: python advect_chanel.py yyyy mm dd')

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
	meteo_path: str,
	output_name: str,
	target_dt: dt.datetime,
	run_hours: int,
) -> None:
	lines = control_path.read_text().splitlines()
	if len(lines) < 6:
		raise ValueError(f'Unexpected CONTROL format in {control_path}.')

	particle_start = 2

	def _find_footer_start_idx() -> int:
		# Scan from bottom and find the last valid CONTROL footer start:
		# run_hours, vertical motion, top, nmet, meteo dir/file pairs, output dir/file.
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

			# Prefer footer whose run_hours matches requested value.
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

	new_particle_lines = [f'{lat:.1f} {lon:.1f} 1000.0' for (lat, lon) in points]

	lines[0] = (
		f'{target_dt.year % 100:02d} {target_dt.month:02d} {target_dt.day:02d} '
		f'{target_dt.hour:02d} {target_dt.minute:02d}'
	)
	lines[1] = str(len(points))
	updated_lines = lines[:particle_start] + new_particle_lines + footer_lines
	new_run_hours_idx = particle_start + len(new_particle_lines)
	updated_lines[new_run_hours_idx] = str(run_hours)

	footer_offset = new_run_hours_idx
	nmet = int(updated_lines[footer_offset + 3].strip())
	first_meteo_dir_idx = footer_offset + 4
	first_meteo_file_idx = footer_offset + 5
	out_name_idx = footer_offset + 4 + 2 * nmet + 1

	updated_lines[first_meteo_dir_idx] = METEO_BASE_DIR
	updated_lines[first_meteo_file_idx] = meteo_path
	updated_lines[out_name_idx] = output_name

	control_path.write_text('\n'.join(updated_lines) + '\n')


def run_hyts(run_dir: Path, executable: Path) -> None:
	if not executable.exists():
		raise FileNotFoundError(f'HYSPLIT executable not found: {executable}')
	log_path = run_dir / 'rsl_out.hysplit'
	with log_path.open('a', encoding='utf-8') as log_file:
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


def particle_fully_within_reference_pairs(
	track_pairs: list[tuple[float, float]],
	reference_pairs: set[tuple[float, float]],
	grid_step: float,
) -> bool:
	for lat, lon in track_pairs:
		key = _snap_latlon_pair(lat, lon, grid_step)
		if key not in reference_pairs:
			return False
	return True


def write_results_csv(rows: list[dict[str, float | int | str]], output_csv: Path) -> None:
	fieldnames = [
		'run_hours',
		'particle_idx',
		'end_lat',
		'end_lon',
		'lat',
		'lon',
	]
	output_csv.parent.mkdir(parents=True, exist_ok=True)
	with output_csv.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def main() -> None:
	target_date = parse_target_date_from_argv(sys.argv)
	source_points = load_points(CSV_PATH, month=target_date.month, count='all')
	land_prepared = build_land_prepared_geometry()
	end_points_all = expand_to_nearby_ocean_points(
		source_points,
		distance_deg=NEARBY_DEGREES,
		land_prepared=land_prepared,
	)

	end_points, selected_start, selected_end = select_particle_range(
		end_points_all,
		start_1based=PARTICLE_INDEX_START,
		end_1based=PARTICLE_INDEX_END,
	)

	points_by_utc = group_points_by_utc_datetime(end_points, target_date)
	utc_slots = build_24_utc_slots_for_lst_date(target_date)
	result_csv_path = RESULT_CSV_DIR / f'affected_latlon_{target_date:%Y%m%d}1330.csv'

	print(f'Source particles in CSV (month={target_date.month}): {len(source_points)}')
	print(f'Expanded particles ({NEARBY_DEGREES}-degree nearby ocean): {len(end_points_all)}')
	print(f'Selected particle range (1-based, inclusive): {selected_start}..{selected_end}')
	print(f'Selected particles to run: {len(end_points)}')
	print(f'Using run_hours={RUN_HOURS}')
	# print('24 UTC slots mapped from LST yyyy-mm-dd 13:30:')
	# for utc_dt in utc_slots:
	# 	print(f'  {utc_dt:%Y-%m-%d %H:%M}')
	reference_pairs = build_pair_set(source_points, PAIR_GRID_STEP)

	saved_rows: list[dict[str, float | int | str]] = []
	total_particles_simulated = 0
	kept_particles = 0
	runs_executed = 0

	for utc_dt in utc_slots:
		hour_points = points_by_utc.get(utc_dt, [])
		if not hour_points:
			continue

		target_dt = utc_dt
		week_index = meteo_week_index_from_date(target_dt.date())
		meteo_path = build_meteo_path(target_dt.year, target_dt.month, week_index)
		output_name = f'output_{target_dt:%Y%m%d}_{target_dt.hour:02d}'

		update_control(CONTROL_PATH, hour_points, meteo_path, output_name, target_dt, RUN_HOURS)
		print(f'T0 datetime (UTC): {target_dt:%Y-%m-%d %H:%M}, particles={len(hour_points)}')

		run_hyts(RUN_DIR, HYSPLIT_EXEC)
		runs_executed += 1

		output_file = RUN_DIR / output_name
		tracks_by_particle = extract_all_particle_latlon_from_output(output_file)
		total_particles_simulated += len(tracks_by_particle)

		end_point_by_particle = {idx + 1: (lat, lon) for idx, (lat, lon) in enumerate(hour_points)}
		for particle_idx in sorted(tracks_by_particle.keys()):
			track = tracks_by_particle[particle_idx]
			if not particle_fully_within_reference_pairs(track, reference_pairs, PAIR_GRID_STEP):
				continue
			kept_particles += 1
			end_lat, end_lon = end_point_by_particle.get(particle_idx, (float('nan'), float('nan')))
			for lat, lon in track:
				saved_rows.append(
					{
						'run_hours': RUN_HOURS,
						'particle_idx': particle_idx,
						'end_lat': float(end_lat),
						'end_lon': float(end_lon),
						'lat': float(lat),
						'lon': float(lon),
					}
				)

	write_results_csv(saved_rows, result_csv_path)
	print(f'Saved CSV: {result_csv_path}')
	print(f'UTC-hour runs executed: {runs_executed}')
	print(f'Kept particles: {kept_particles}/{total_particles_simulated}')
	print(f'Rows: {len(saved_rows)}')


if __name__ == '__main__':
	main()
