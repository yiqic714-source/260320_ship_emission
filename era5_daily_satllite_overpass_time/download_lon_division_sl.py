import numpy as np
import cdsapi
import os  # 用于文件检测和创建文件夹

target_year = 2010
satellite = "Aqua"    # "Aqua" 或 "Terra"
# ==============================================================

dataset = "reanalysis-era5-single-levels"
client = cdsapi.Client()

if satellite == "Terra":
    local_t = 10.5
elif satellite == "Aqua":
    local_t = 13.5

for utc_hour in range(0, 24):
    lon_min = 15 * (local_t - (utc_hour + 0.5))
    lon_max = 15 * (local_t - (utc_hour - 0.5))
    if lon_min > 180:
        lon_min = lon_min - 360 
    elif lon_min < -180:
        lon_min = lon_min + 360 

    if lon_max > 180:
        lon_max = lon_max - 360 
    elif lon_max < -180:
        lon_max = lon_max + 360

    if abs(lon_min) == 180:
        lon_min = 180 * lon_max/abs(lon_max)
    elif abs(lon_max) == 180:
        lon_max = 180 * lon_min/abs(lon_min)
    
    lon_max = lon_max - 0.1
    
    print(f"UTC {utc_hour:02d}:00 → 对应经度范围：{lon_min:.1f}° ~ {lon_max:.1f}°") 

    # 遍历月份
    for month in range(1, 13):
        # 文件名使用自定义年份
        output_filename = f"{target_year}_LST1330_sl/era5_sl_{target_year}{month:02d}_utc{utc_hour:02d}.nc"

        # ====================== 【核心：检测文件是否存在】 ======================
        if os.path.exists(output_filename):
            print(f"✅ 文件已存在，跳过：{output_filename}")
            continue  # 存在则直接跳过，不下载
        # ========================================================================

        # 自动创建文件夹（防止报错）
        os.makedirs(os.path.dirname(output_filename), exist_ok=True)

        request = {
            "product_type": ["reanalysis"],
            "variable": [
                "10m_u_component_of_wind",
                "10m_v_component_of_wind",
                "2m_dewpoint_temperature",
                "2m_temperature",
                "mean_sea_level_pressure",
                "mean_wave_direction",
                "mean_wave_period",
                "sea_surface_temperature",
                "surface_pressure",
                "total_precipitation",
                "large_scale_precipitation",
                "large_scale_precipitation_fraction",
                "precipitation_type",
                "boundary_layer_dissipation",
                "boundary_layer_height",
                "convective_available_potential_energy"
            ],
            "year": [str(target_year)],  # 使用自定义年份
            "month": [f"{month:02d}"],
            "day": [f"{d:02d}" for d in range(1, 32)],  # 简化写法
            "time": [f"{utc_hour:02d}:00"],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [60, lon_min, -60, lon_max]
        }

        print(f"🔽 开始下载：{output_filename}")
        client.retrieve(dataset, request, output_filename)