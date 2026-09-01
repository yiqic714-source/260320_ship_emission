import numpy as np
import cdsapi
import os
import argparse
import calendar

# ============================================================
# 运行方式：
#   python xxx.py 2015
#   python xxx.py 2015 --satellite Aqua
#   python xxx.py 2015 --satellite Terra
# ============================================================

parser = argparse.ArgumentParser(description="Download ERA5 pressure-level data by year.")
parser.add_argument("year", type=int, help="Target year, e.g., 2015")
parser.add_argument(
    "--satellite",
    type=str,
    default="Aqua",
    choices=["Aqua", "Terra"],
    help="Satellite local overpass time: Aqua=13:30, Terra=10:30"
)

args = parser.parse_args()

target_year = args.year
satellite = args.satellite

dataset = "reanalysis-era5-pressure-levels"
client = cdsapi.Client()

# 本地时
if satellite == "Terra":
    local_t = 10.5
    lst_label = "LST1030"
elif satellite == "Aqua":
    local_t = 13.5
    lst_label = "LST1330"

# 输出文件夹
out_dir = f"{target_year}_{lst_label}_pl"
os.makedirs(out_dir, exist_ok=True)

# 遍历24个UTC时刻
for utc_hour in range(0, 24):

    # 计算经度范围
    lon_min = 15 * (local_t - (utc_hour + 0.5))
    lon_max = 15 * (local_t - (utc_hour - 0.5))

    if lon_min > 180:
        lon_min -= 360
    elif lon_min < -180:
        lon_min += 360

    if lon_max > 180:
        lon_max -= 360
    elif lon_max < -180:
        lon_max += 360

    if abs(lon_min) == 180:
        lon_min = 180 * lon_max / abs(lon_max)
    elif abs(lon_max) == 180:
        lon_max = 180 * lon_min / abs(lon_min)

    lon_max = lon_max - 0.1

    print(f"UTC {utc_hour:02d}:00 → 对应经度范围：{lon_min:.1f}° ~ {lon_max:.1f}°")

    # 月份遍历
    for month in range(1, 13):

        output_filename = os.path.join(
            out_dir,
            f"era5_pl_{target_year}{month:02d}_utc{utc_hour:02d}.nc"
        )

        # ====================================================
        # 文件存在且非空，则跳过
        # ====================================================
        if os.path.exists(output_filename) and os.path.getsize(output_filename) > 0:
            print(f"✅ 文件已存在，跳过下载：{output_filename}")
            continue

        # 如果文件存在但大小为0，说明可能是上次下载失败，删除后重下
        if os.path.exists(output_filename) and os.path.getsize(output_filename) == 0:
            print(f"⚠️ 文件为空，删除后重新下载：{output_filename}")
            os.remove(output_filename)

        # 当前月份真实天数，避免2月、4月、6月等月份请求不存在日期
        _, ndays = calendar.monthrange(target_year, month)
        days = [f"{d:02d}" for d in range(1, ndays + 1)]

        request = {
            "product_type": ["reanalysis"],
            "variable": [
                "divergence",
                "geopotential",
                "potential_vorticity",
                "relative_humidity",
                "specific_humidity",
                "temperature",
                "u_component_of_wind",
                "v_component_of_wind",
                "vertical_velocity",
                "vorticity"
            ],
            "pressure_level": ["500", "650", "750", "850", "925", "1000"],
            "year": [str(target_year)],
            "month": f"{month:02d}",
            "day": days,
            "time": [f"{utc_hour:02d}:00"],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [60, lon_min, -60, lon_max]
        }

        print(f"🔽 开始下载：{output_filename}")

        client.retrieve(dataset, request, output_filename)

print("全部任务完成。")