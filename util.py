import datetime as dt
import math


def round_to_nearest_utc_hour(time_value: dt.datetime) -> dt.datetime:
	base = time_value.replace(minute=0, second=0, microsecond=0)
	delta_seconds = (time_value - base).total_seconds()
	if delta_seconds > 1800:
		base += dt.timedelta(hours=1)
	return base


def nearest_utc_for_satellite_overpass_time(
	lst_date: dt.date,
	lon: float,
	target_lst_hour: float,
) -> dt.datetime:
	lst_hour = int(math.floor(target_lst_hour))
	lst_minute = int(round((target_lst_hour - lst_hour) * 60.0))
	lst_dt = dt.datetime(lst_date.year, lst_date.month, lst_date.day, lst_hour, lst_minute)
	utc_dt = lst_dt - dt.timedelta(hours=(lon / 15.0))
	return round_to_nearest_utc_hour(utc_dt)


def lon_to_utc_hour(lon: float, target_lst_hour: float) -> int:
	ref_date = dt.date(2000, 1, 1)
	utc_dt = nearest_utc_for_satellite_overpass_time(
		lst_date=ref_date,
		lon=lon,
		target_lst_hour=target_lst_hour,
	)
	return int(utc_dt.hour)
