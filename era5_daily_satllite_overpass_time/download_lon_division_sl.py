import numpy as np
import cdsapi
dataset = "reanalysis-era5-single-levels"
client = cdsapi.Client()
satellite = "Aqua"#"Terra"

if satellite=="Terra":
    local_t = 10.5
elif satellite=="Aqua":
    local_t = 13.5

for utc_hour in range(24):
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

    for month in range(1, 13):
        output_filename = f"2020_LST1330_sl/era5_sl_2020{month:02d}_utc{utc_hour:02d}.nc"

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
            "year": ["2019"],
            "month": [f"{month:02d}"],
            "day": [
                "01", "02", "03",
                "04", "05", "06",
                "07", "08", "09",
                "10", "11", "12",
                "13", "14", "15",
                "16", "17", "18",
                "19", "20", "21",
                "22", "23", "24",
                "25", "26", "27",
                "28", "29", "30",
                "31"
            ],
            "time": [
                f"{utc_hour:02d}:00"
            ],
            "data_format": "netcdf",
            "download_format": "unarchived",
            "area": [60, lon_min, -60, lon_max]
        }

        client.retrieve(dataset, request, output_filename)
