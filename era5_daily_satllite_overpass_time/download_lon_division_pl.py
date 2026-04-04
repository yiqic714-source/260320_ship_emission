import numpy as np
import cdsapi
dataset = "reanalysis-era5-pressure-levels"
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
        output_filename = f"2019_LST1330_pl/era5_pl_2019{month:02d}_utc{utc_hour:02d}.nc"

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
            "pressure_level": [
                "500", "650", "750",
                "850", "925", "1000"
            ],
            "year": ["2019"],
            "month": month,
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