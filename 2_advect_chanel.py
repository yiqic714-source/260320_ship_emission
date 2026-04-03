from __future__ import annotations

import csv
import datetime as dt
import subprocess
import sys
from pathlib import Path


CSV_PATH = Path('/home/chenyiqi/260320_ship_emission/processed_data/sox_selected_points_2019_minus_2020_top10.csv')
CONTROL_PATH = Path('/home/chenyiqi/hysplit/hysplit.v5.4.2_RHEL8.10_public/test_advect_chanel/CONTROL')
RUN_DIR = CONTROL_PATH.parent
HYSPLIT_EXEC = RUN_DIR.parent / 'exec' / 'hyts_std'

TARGET_HOUR = 14
TARGET_MINUTE = 0
MAX_BACK_HOURS = 16
# Set an integer to take top-N points, or set to None / 'all' to use all points.
TOP_POINT_COUNT = 'all'
RESULT_CSV_DIR = Path('/home/chenyiqi/260320_ship_emission/processed_data/hysplit_rsl')

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
		raise ValueError('TOP_POINT_COUNT must be a positive integer, None, or "all".')

	points: list[tuple[float, float]] = []
	with csv_path.open(newline='') as handle:
		reader = csv.DictReader(handle)
		for row in reader:
			if int(row['month']) != month:
				continue
			points.append((float(row['lat']), float(row['lon'])))
			if not use_all and len(points) == count:
				break

	if not use_all and len(points) < count:
		raise ValueError(f'Only found {len(points)} points for month={month}, need {count}.')
	if use_all and not points:
		raise ValueError(f'No points found for month={month}.')
	return points


def meteo_week_index_from_date(date_value: dt.date) -> int:
	# GDAS weekly file convention: w1=days 1-7, w2=8-14, w3=15-21, w4=22-28, w5=29-end.
	return ((date_value.day - 1) // 7) + 1


def build_meteo_path(year: int, month: int, week_index: int) -> str:
	month_abbr = MONTH_ABBR[month]
	return f'{year}/gdas1.{month_abbr}{year % 100:02d}.w{week_index}'


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

	particle_count = int(lines[1].strip())
	particle_start = 2
	particle_end = particle_start + particle_count
	if len(lines) <= particle_end + 3:
		raise ValueError(f'CONTROL file is shorter than expected: {control_path}.')

	heights = []
	for line in lines[particle_start:particle_end]:
		parts = line.split()
		if len(parts) < 3:
			raise ValueError(f'Invalid particle line: {line!r}')
		heights.append(parts[2])

	if not heights:
		raise ValueError('No particle heights found in CONTROL.')

	if len(heights) < len(points):
		heights.extend([heights[-1]] * (len(points) - len(heights)))

	new_particle_lines = [
		f'{lat:.1f} {lon:.1f} {height}'
		for (lat, lon), height in zip(points, heights)
	]

	lines[0] = (
		f'{target_dt.year % 100:02d} {target_dt.month:02d} {target_dt.day:02d} '
		f'{target_dt.hour:02d} {target_dt.minute:02d}'
	)
	lines[1] = str(len(points))
	updated_lines = lines[:particle_start] + new_particle_lines + lines[particle_end:]
	updated_lines[particle_end] = str(run_hours)
	updated_lines[-3] = meteo_path
	updated_lines[-1] = output_name

	control_path.write_text('\n'.join(updated_lines) + '\n')


def run_hyts(run_dir: Path, executable: Path) -> None:
	if not executable.exists():
		raise FileNotFoundError(f'HYSPLIT executable not found: {executable}')
	log_path = run_dir / 'rsl_out.hysplit'
	with log_path.open('a', encoding='utf-8') as log_file:
		subprocess.run([str(executable)], cwd=run_dir, check=True, stdout=log_file, stderr=log_file)


def extract_target_latlon_from_output(output_file: Path, target_dt: dt.datetime) -> list[dict[str, float | int | str]]:
	rows: list[dict[str, float | int | str]] = []
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

		year = int(parts[2]) + 2000
		month = int(parts[3])
		day = int(parts[4])
		hour = int(parts[5])
		minute = int(parts[6])

		if (
			year == target_dt.year
			and month == target_dt.month
			and day == target_dt.day
			and hour == target_dt.hour
			and minute == target_dt.minute
		):
			rows.append(
				{
					'particle': int(parts[0]),
					'lat': float(parts[9]),
					'lon': float(parts[10]),
				}
			)

	if not rows:
		raise ValueError(f'No records found in {output_file} for target datetime {target_dt:%Y-%m-%d %H:%M}.')

	rows.sort(key=lambda x: int(x['particle']))
	return rows


def write_results_csv(rows: list[dict[str, float | int | str]], output_csv: Path) -> None:
	fieldnames = [
		'run_hours',
		'particle',
		'start_lat',
		'start_lon',
		'lat',
		'lon',
	]
	with output_csv.open('w', newline='', encoding='utf-8') as handle:
		writer = csv.DictWriter(handle, fieldnames=fieldnames)
		writer.writeheader()
		writer.writerows(rows)


def main() -> None:
	target_date = parse_target_date_from_argv(sys.argv)
	target_dt = dt.datetime(
		target_date.year,
		target_date.month,
		target_date.day,
		TARGET_HOUR,
		TARGET_MINUTE,
	)
	output_name = f'output_{target_dt:%y%m%d%H}'
	result_csv_path = RESULT_CSV_DIR / f'affected_latlon_{target_dt:%y%m%d%H}.csv'

	all_rows: list[dict[str, float | int | str]] = []
	for run_hours in range(1, MAX_BACK_HOURS + 1):
		start_dt = target_dt - dt.timedelta(hours=run_hours)
		points = load_points(CSV_PATH, month=start_dt.month, count=TOP_POINT_COUNT)
		start_point_by_particle = {
			idx + 1: (point_lat, point_lon)
			for idx, (point_lat, point_lon) in enumerate(points)
		}
		week_index = meteo_week_index_from_date(start_dt.date())
		meteo_path = build_meteo_path(start_dt.year, start_dt.month, week_index)

		update_control(CONTROL_PATH, points, meteo_path, output_name, start_dt, run_hours)
		print(f'Start datetime: {start_dt:%Y-%m-%d %H:%M}, run_hours={run_hours}')
		# print(f'Meteorology file: {meteo_path}')

		run_hyts(RUN_DIR, HYSPLIT_EXEC)

		output_file = RUN_DIR / output_name
		target_rows = extract_target_latlon_from_output(output_file, target_dt)
		for row in target_rows:
			particle_id = int(row['particle'])
			start_lat, start_lon = start_point_by_particle.get(particle_id, (float('nan'), float('nan')))
			all_rows.append(
				{
					'run_hours': run_hours,
					'particle': particle_id,
					'start_lat': float(start_lat),
					'start_lon': float(start_lon),
					'lat': float(row['lat']),
					'lon': float(row['lon']),
				}
			)

	write_results_csv(all_rows, result_csv_path)
	print(f'Saved CSV: {result_csv_path}')
	print(f'Rows: {len(all_rows)}')


if __name__ == '__main__':
	main()
