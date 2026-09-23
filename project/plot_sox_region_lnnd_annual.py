"""
绘制「accu_sox 分位点分档区域」lnnd 年均值的年际变化（per-sample 口径）。

口径（分位点取法与 unet_finetune_pretrained.build_mask_from_accu_sox() 一致）：
    1. 每个日样本（天）用自己的 accu_sox 排序，在「当天 sox 有效 且 lnnd 有效」
       的格点中按分位点切成若干条带（默认 0-20%、20-40%、40-60%、60-80%、80-100%），
       每条带即当天的「该 SOx 强度区间」区域；
    2. 对每条带内的 lnnd 做 cos(lat) 面积加权平均，得到当天该条带的区域平均 lnnd；
    3. 对年内的天取平均，得到该年该条带的年均值；
    4. 所有条带的年际变化曲线画在同一张图上。

注意：区域会逐日/逐月移动（accu_sox 是月累积场，月内每天相同、月际变化）。

用法：
    conda activate my_env
    cd /home/chenyiqi/260320_ship_emission/project
    python plot_sox_region_lnnd_annual.py

输出（OUT_DIR 下，文件名带 OUTPUT_TAG）：
    lnnd_annual_{OUTPUT_TAG}.png   各分位带的年际变化曲线
    lnnd_annual_{OUTPUT_TAG}.csv   逐年的各条带区域平均 lnnd，便于后续画图/统计
"""

from __future__ import annotations

import csv
import os
from pathlib import Path

if not os.environ.get('DISPLAY'):
    # 无显示环境下使用不弹窗的后端，保证能保存图片。
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr

from util import collect_grid_files


# ------------------------------------------------------------
# 输入 / 输出
# ------------------------------------------------------------
DATA_ROOT = Path('/home/chenyiqi/260320_ship_emission/processed_data/ml_grid_data')
OUT_DIR = Path('/home/chenyiqi/260320_ship_emission/project/figs')

# 输出文件名后缀，方便区分不同参数的实验。
OUTPUT_TAG = 'sox_quintiles_persample'

# 参与统计的年份。
YEARS = tuple(range(2005, 2023))

# 每个样本内按 accu_sox 分位点切成若干条带，逐条带求区域平均。
# 默认 5 档：q0-20%、q20-40%、q40-60%、q60-80%、q80-100%。
QUANTILE_EDGES = (0.0, 0.2, 0.4, 0.6, 0.8, 1.0)

QUANTILE_BANDS = tuple(
    (QUANTILE_EDGES[index], QUANTILE_EDGES[index + 1])
    for index in range(len(QUANTILE_EDGES) - 1)
)


# ------------------------------------------------------------
# 计算部分
# ------------------------------------------------------------
def band_label(
    start_quantile: float,
    end_quantile: float,
) -> str:
    """图例用的条带标签，例如 'q0-20%'。"""
    return f'q{100 * start_quantile:.0f}-{100 * end_quantile:.0f}%'


def band_column_name(
    start_quantile: float,
    end_quantile: float,
) -> str:
    """CSV 列名，例如 'lnnd_annual_q0_20'。"""
    return (
        f'lnnd_annual_q{100 * start_quantile:.0f}'
        f'_{100 * end_quantile:.0f}'
    )


def quantile_band_slice(
    n_valid: int,
    start_quantile: float,
    end_quantile: float,
) -> tuple[int, int]:
    """
    把「分位点区间」换算成升序排序后的下标区间 [start, end)。

    与 unet_finetune_pretrained.build_mask_from_accu_sox() 的取法一致：
    start = floor(start_quantile * n)，end = ceil(end_quantile * n)，
    再做 clamp，保证至少取到 1 个格点。
    """
    start = int(np.floor(start_quantile * n_valid))
    end = int(np.ceil(end_quantile * n_valid))
    start = min(max(start, 0), n_valid - 1)
    end = min(max(end, start + 1), n_valid)
    return start, end


def area_weighted_mean_at_cells(
    field_flat: np.ndarray,
    weight_flat: np.ndarray,
    cells: np.ndarray,
) -> tuple[float, int]:
    """
    在给定格点（扁平下标，调用方已保证 field 有效）上做面积加权平均。

    返回 (加权平均, 格点数)。
    """
    if cells.size == 0:
        return float('nan'), 0

    weights = weight_flat[cells]
    total = float(np.sum(weights))
    if total <= 0.0:
        return float('nan'), 0

    return float(np.sum(field_flat[cells] * weights) / total), int(cells.size)


def get_grid_coordinates(sample_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(sample_path) as ds:
        lat = ds['lat'].values.astype(np.float64)
        lon = ds['lon'].values.astype(np.float64)
    return lat, lon


def accumulate_band_means(
    paths: list[Path],
    years: tuple[int, ...],
    area_weight: np.ndarray,
    bands: tuple[tuple[float, float], ...],
) -> dict:
    """
    逐个日样本计算每条分位带内的面积加权平均 lnnd，并按年累加（每天等权）。

    返回：
        band_sum   : {year: ndarray(n_bands)} 每年各条带「当天区域平均」之和
        day_count  : {year: int}              每年用到的天数
        cell_sum   : {year: ndarray(n_bands)} 每年各条带用到的格点数之和
    """
    n_bands = len(bands)
    shape = area_weight.shape
    weight_flat = area_weight.ravel()

    band_sum = {
        year: np.zeros(n_bands, dtype=np.float64)
        for year in years
    }
    day_count = {year: 0 for year in years}
    cell_sum = {
        year: np.zeros(n_bands, dtype=np.float64)
        for year in years
    }

    total = len(paths)
    for index, path in enumerate(paths, start=1):
        with xr.open_dataset(path) as ds:
            year = int(ds.attrs['year'])
            sox = ds['accu_sox'].values.astype(np.float64)
            lnnd = ds['lnnd'].values.astype(np.float64)

        sox_flat = sox.ravel()
        lnnd_flat = lnnd.ravel()

        # 与 build_mask_from_accu_sox() 一致：候选格点 = sox 有效 & 目标有效。
        candidate = np.isfinite(sox_flat) & np.isfinite(lnnd_flat)
        flat_idx = np.flatnonzero(candidate)
        n_valid = flat_idx.size
        if n_valid == 0:
            continue

        # 每个样本只排序一次，之后按分位点区间切成各条带。
        order = np.argsort(sox_flat[flat_idx])

        day_means = np.full(n_bands, np.nan, dtype=np.float64)
        day_cells = np.zeros(n_bands, dtype=np.int64)

        for band_index, (start_q, end_q) in enumerate(bands):
            start, end = quantile_band_slice(
                n_valid,
                start_q,
                end_q,
            )
            cells = flat_idx[order[start:end]]
            band_mean, n_cells = area_weighted_mean_at_cells(
                lnnd_flat,
                weight_flat,
                cells,
            )
            day_means[band_index] = band_mean
            day_cells[band_index] = n_cells

        if not np.isfinite(day_means).all():
            continue

        band_sum[year] += day_means
        cell_sum[year] += day_cells
        day_count[year] += 1

        if index % 500 == 0 or index == total:
            print(f'  processed {index}/{total} files', flush=True)

    return {
        'band_sum': band_sum,
        'day_count': day_count,
        'cell_sum': cell_sum,
    }


def annual_mean_series(
    accumulated: dict,
    years: tuple[int, ...],
    n_bands: int,
) -> tuple[dict[int, np.ndarray], dict[int, int], dict[int, np.ndarray]]:
    """
    把逐日各条带的区域平均换算成逐年年均值。

    返回 (年均值 {year: ndarray(n_bands)}, 该年天数, 平均每天各条带格点数)。
    """
    band_sum = accumulated['band_sum']
    day_count = accumulated['day_count']
    cell_sum = accumulated['cell_sum']

    annual = {}
    days = {}
    mean_cells = {}

    for year in years:
        n_days = day_count[year]
        days[year] = n_days
        if n_days > 0:
            annual[year] = band_sum[year] / n_days
            mean_cells[year] = cell_sum[year] / n_days
        else:
            annual[year] = np.full(n_bands, np.nan, dtype=np.float64)
            mean_cells[year] = np.full(n_bands, np.nan, dtype=np.float64)

    return annual, days, mean_cells


def write_csv(
    csv_path: Path,
    years: tuple[int, ...],
    annual: dict[int, np.ndarray],
    days: dict[int, int],
    bands: tuple[tuple[float, float], ...],
) -> None:
    with csv_path.open('w', newline='', encoding='utf-8') as handle:
        writer = csv.writer(handle)
        writer.writerow(
            ['year']
            + [
                band_column_name(start_q, end_q)
                for start_q, end_q in bands
            ]
            + ['days_used']
        )
        for year in years:
            writer.writerow(
                [year]
                + [f'{value:.6f}' for value in annual[year]]
                + [days[year]]
            )



# ------------------------------------------------------------
# 画图 / 主流程
# ------------------------------------------------------------
def plot_annual_series(
    figure_path: Path,
    years: tuple[int, ...],
    annual: dict[int, np.ndarray],
    bands: tuple[tuple[float, float], ...],
) -> None:
    """把各分位带的年均值年际变化画在同一张图上。"""
    x = np.asarray(years, dtype=np.float64)
    colours = plt.get_cmap('viridis')(
        np.linspace(0.0, 1.0, len(bands))
    )

    fig, ax_series = plt.subplots(
        figsize=(9, 5),
        constrained_layout=True,
    )

    for band_index, (start_q, end_q) in enumerate(bands):
        y = np.asarray(
            [annual[year][band_index] for year in years],
            dtype=np.float64,
        )
        ax_series.plot(
            x,
            y,
            'o-',
            color=colours[band_index],
            label=band_label(start_q, end_q),
        )

    ax_series.set_xlabel('Year')
    ax_series.set_ylabel('Annual mean lnNd')
    ax_series.set_title(
        'Interannual variation of annual mean lnNd\n'
        'by accu_sox quantile band '
        '(per-sample bands, cos(lat) area-weighted)'
    )
    ax_series.grid(alpha=0.3)
    ax_series.legend(
        loc='best',
        fontsize=9,
        title='accu_sox quantile band',
    )

    fig.savefig(figure_path, dpi=250, bbox_inches='tight')
    plt.close(fig)


def main() -> None:
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    paths = collect_grid_files(DATA_ROOT, YEARS)

    print(f'Years         : {YEARS[0]}-{YEARS[-1]}')
    print(
        'Quantile bands: '
        + ', '.join(
            band_label(start_q, end_q)
            for start_q, end_q in QUANTILE_BANDS
        )
    )
    print('Region mask   : per-sample (each day uses its own accu_sox)')
    print(f'Files         : {len(paths)}')

    lat, lon = get_grid_coordinates(paths[0])
    area_weight = (
        np.cos(np.deg2rad(lat))[:, None]
        * np.ones((1, lon.size), dtype=np.float64)
    )

    print('Accumulating daily band means...')
    accumulated = accumulate_band_means(
        paths,
        YEARS,
        area_weight,
        QUANTILE_BANDS,
    )
    annual, days, mean_cells = annual_mean_series(
        accumulated,
        YEARS,
        len(QUANTILE_BANDS),
    )

    print()
    print(
        '  year   '
        + '  '.join(
            f'{band_label(start_q, end_q):>9}'
            for start_q, end_q in QUANTILE_BANDS
        )
        + '   days'
    )
    for year in YEARS:
        print(
            f'  {year}   '
            + '  '.join(
                f'{value:9.4f}' for value in annual[year]
            )
            + f'   {days[year]:4d}'
        )

    print()
    print('Mean region cells per band:')
    for band_index, (start_q, end_q) in enumerate(QUANTILE_BANDS):
        band_cells = float(np.nanmean([
            mean_cells[year][band_index]
            for year in YEARS
        ]))
        print(
            f'  {band_label(start_q, end_q):>9}: '
            f'{band_cells:8.1f} cells'
        )

    csv_path = OUT_DIR / f'lnnd_annual_{OUTPUT_TAG}.csv'
    figure_path = OUT_DIR / f'lnnd_annual_{OUTPUT_TAG}.png'
    write_csv(csv_path, YEARS, annual, days, QUANTILE_BANDS)
    plot_annual_series(
        figure_path,
        YEARS,
        annual,
        QUANTILE_BANDS,
    )

    print()
    print(f'Saved CSV   : {csv_path}')
    print(f'Saved figure: {figure_path}')


if __name__ == '__main__':
    main()

