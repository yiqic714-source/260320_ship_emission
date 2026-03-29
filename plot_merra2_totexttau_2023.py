import glob
import os
import numpy as np
import netCDF4 as nc
import matplotlib.pyplot as plt


def get_lon_mask_for_utc_hour(local_t, utc_hour, lon):
    """Calculate longitude mask for specific UTC hour and local time."""
    lon_min = 15 * (local_t - (utc_hour + 0.5))
    lon_max = 15 * (local_t - (utc_hour - 0.5)) - 0.1

    # Keep identical edge handling as the original workflow.
    if utc_hour == 22.5:
        lon_mask = ((lon >= -180) & (lon <= -172.6)) | ((lon >= 172.5) & (lon <= 180))
    elif utc_hour == 23.5:
        lon_mask = (lon >= 157.5) & (lon <= 172.4)
    else:
        lon_mask = (lon >= lon_min) & (lon <= lon_max)

    return lon_mask


def update_daily_localtime_totexttau(file_path, local_t, annual_sum, annual_count):
    """Read one MERRA2 file and accumulate local-time TOTEXTTAU into annual arrays."""
    with nc.Dataset(file_path, "r") as ds:
        lon = ds.variables["lon"][:].astype(np.float32)
        totexttau = ds.variables["TOTEXTTAU"][:].astype(np.float32)  # [time, lat, lon]

        for utc_hour in np.arange(0.5, 24.0, 1.0):
            lon_mask = get_lon_mask_for_utc_hour(local_t, utc_hour, lon)
            lon_idx = np.where(lon_mask)[0]
            if lon_idx.size == 0:
                continue

            t_idx = int(utc_hour - 0.5)
            # Use np.take on lon axis to keep output strictly [lat, selected_lon].
            ttau_slice = np.take(totexttau[t_idx, :, :], lon_idx, axis=1)
            valid = np.isfinite(ttau_slice)

            annual_sum[:, lon_idx] += np.where(valid, ttau_slice, 0.0)
            annual_count[:, lon_idx] += valid.astype(np.int32)


def compute_annual_mean_totexttau(data_dir, year, local_t=10.5):
    """Compute annual mean TOTEXTTAU at target local time for one year."""
    pattern = os.path.join(data_dir, f"MERRA2_*00.tavg1_2d_aer_Nx.{year}*.SUB.nc")
    files = sorted(glob.glob(pattern))
    if not files:
        raise FileNotFoundError(f"No files found with pattern: {pattern}")

    with nc.Dataset(files[0], "r") as ds0:
        lon = ds0.variables["lon"][:].astype(np.float32)
        lat = ds0.variables["lat"][:].astype(np.float32)

    annual_sum = np.zeros((lat.size, lon.size), dtype=np.float64)
    annual_count = np.zeros((lat.size, lon.size), dtype=np.int32)

    print(f"Found {len(files)} files for {year}")
    for i, file_path in enumerate(files, start=1):
        if i == 1 or i % 50 == 0 or i == len(files):
            print(f"Processing {i}/{len(files)}: {os.path.basename(file_path)}")
        update_daily_localtime_totexttau(file_path, local_t, annual_sum, annual_count)

    with np.errstate(divide="ignore", invalid="ignore"):
        annual_mean = np.where(annual_count > 0, annual_sum / annual_count, np.nan).astype(np.float32)

    return lon, lat, annual_mean


def plot_global_totexttau(lon, lat, annual_mean, output_png, year):
    """Plot global annual mean TOTEXTTAU distribution."""
    lon2d, lat2d = np.meshgrid(lon, lat)

    finite_vals = annual_mean[np.isfinite(annual_mean)]
    if finite_vals.size == 0:
        raise ValueError("No valid values available for plotting.")

    vmin = np.percentile(finite_vals, 1)
    vmax = np.percentile(finite_vals, 99)

    fig, ax = plt.subplots(figsize=(14, 6), dpi=300)
    mesh = ax.pcolormesh(lon2d, lat2d, annual_mean, shading="auto", cmap="viridis", vmin=vmin, vmax=vmax)

    ax.set_xlim(-180, 180)
    ax.set_ylim(-90, 90)
    ax.set_xticks(np.arange(-180, 181, 60))
    ax.set_yticks(np.arange(-90, 91, 30))
    ax.set_xlabel("Longitude")
    ax.set_ylabel("Latitude")
    ax.set_title(f"MERRA2 TOTEXTTAU Annual Mean ({year}, Local Time 10:30)")

    cbar = fig.colorbar(mesh, ax=ax, orientation="horizontal", pad=0.08, aspect=60)
    cbar.set_label("TOTEXTTAU")

    fig.tight_layout()
    fig.savefig(output_png, bbox_inches="tight")
    plt.close(fig)

    print(f"Saved plot: {output_png}")


def main():
    year = "2023"
    local_t = 10.5
    data_dir = f"/home/chenyiqi/251201_ERFaci/merra2_hourly/{year}"
    output_dir = "/home/chenyiqi/260320_ship_emission/figs"
    os.makedirs(output_dir, exist_ok=True)

    output_png = os.path.join(output_dir, f"merra2_totexttau_{year}_global.png")

    lon, lat, annual_mean = compute_annual_mean_totexttau(data_dir, year, local_t=local_t)
    plot_global_totexttau(lon, lat, annual_mean, output_png, year)


if __name__ == "__main__":
    main()
