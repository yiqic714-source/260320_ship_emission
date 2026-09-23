from pathlib import Path
import datetime as dt
import json
import os
import matplotlib.pyplot as plt
import numpy as np
import re
import torch
from typing import Sequence
import torch.nn as nn
import xarray as xr
from torch.utils.data import DataLoader
from scipy.stats import t as student_t

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature
    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False

MASK_DEFINITION = ""
VALIDATION_YEARS: Sequence[int] = ()
COUNTERFACTUAL_YEARS: Sequence[int] = ()
OUTPUT_TAG = 'output'


def normalize_years(years: int | Sequence[int]) -> tuple[int, ...]:
    """
    年份参数归一化：单个年份和年份序列都接受。

    Python 里 (2019) 只是 int，单元素元组必须写成 (2019,)；
    这里两种写法都兼容，避免出现
    `TypeError: argument of type 'int' is not iterable`。
    """
    if isinstance(years, str):
        raise TypeError(
            'years must be an int or a sequence of ints, '
            f'got {years!r}'
        )

    if isinstance(years, (int, np.integer)):
        return (int(years),)

    return tuple(int(year) for year in years)


def parse_date_from_path(path: Path) -> dt.date | None:
    match = re.search(r'_(\d{8})\d{4}\.nc$', path.name)
    if not match:
        return None
    return dt.datetime.strptime(match.group(1), '%Y%m%d').date()


def collect_grid_files(
    data_root: Path,
    years: int | tuple[int, ...],
) -> list[Path]:
    """
    收集指定年份的日网格文件（按文件名里的日期过滤）。

    单个年份（如 2019）或年份序列（如 (2018, 2019)）都接受。
    """
    year_set = set(normalize_years(years))

    paths = []
    for path in sorted(data_root.glob('*/ml_grid_*.nc')):
        date_value = parse_date_from_path(path)
        if date_value is not None and date_value.year in year_set:
            paths.append(path)
    if not paths:
        raise ValueError(
            f'No daily grid files found in {data_root} '
            f'for years={sorted(year_set)}'
        )
    return paths


def configure_output(mask_definition: str, validation_years: Sequence[int], counterfactual_years: Sequence[int], output_tag: str) -> None:
    global MASK_DEFINITION, VALIDATION_YEARS, COUNTERFACTUAL_YEARS, OUTPUT_TAG
    MASK_DEFINITION = mask_definition
    VALIDATION_YEARS = normalize_years(validation_years)
    COUNTERFACTUAL_YEARS = normalize_years(counterfactual_years)
    OUTPUT_TAG = output_tag

def save_json(output_path: Path, data: object) -> None:
    output_path.write_text(
        json.dumps(data, indent=2),
        encoding='utf-8',
    )


# ------------------------------------------------------------
# Loss functions shared by pretrain_random_mask.py and
# unet_finetune_pretrained.py. 两个脚本共用同一份实现，
# 修改这里两边同时生效。
# ------------------------------------------------------------
SUPPORTED_LOSS_TYPES = ('mse', 'mae')


def validate_loss_type(loss_type: str) -> str:
    """
    校验 LOSS_TYPE 取值，返回规范化后的小写名称。

    'mse' -> 均方误差
    'mae' -> 平均绝对误差
    """
    normalized = str(loss_type).strip().lower()
    if normalized not in SUPPORTED_LOSS_TYPES:
        raise ValueError(
            'LOSS_TYPE must be one of '
            f'{SUPPORTED_LOSS_TYPES}, got {loss_type!r}'
        )
    return normalized


def masked_loss(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
    loss_type: str,
) -> torch.Tensor:
    """
    带 mask 的逐格点损失，只在 mask == 1 的格点上取平均。

    loss_type='mse' -> mean((pred - target) ** 2)
    loss_type='mae' -> mean(|pred - target|)

    mask 全为 0 时用 clamp 避免除零，返回 0。
    """
    normalized = validate_loss_type(loss_type)

    denom = torch.clamp(mask.sum(), min=1.0)
    residual = pred - target

    if normalized == 'mse':
        per_cell = residual ** 2
    else:
        per_cell = residual.abs()

    return torch.sum(per_cell * mask) / denom


def _empty_spatial_accumulator(
    shape: tuple[int, int],
) -> dict[str, np.ndarray]:
    return {
        'count': np.zeros(shape, dtype=np.int64),
        'sum': np.zeros(shape, dtype=np.float64),
        'sum_sq': np.zeros(shape, dtype=np.float64),
        'abs_sum': np.zeros(shape, dtype=np.float64),
        'abs_sum_sq': np.zeros(shape, dtype=np.float64),
    }


@torch.no_grad()
def collect_spatial_residual_stats(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, np.ndarray]:
    """
    对 unknown/high-SOx 区域累计 residual = observed - reconstructed。

    对验证期：
        residual = 验证预测误差

    对 COUNTERFACTUAL_YEARS：
        residual = lnNd_observed - lnNd_counterfactual
    """
    model.eval()
    accumulator = None

    for batch in loader:
        x = batch['x'].to(device, non_blocking=True)
        y = batch['y'].to(device, non_blocking=True)
        unknown_mask = batch['unknown_mask'].to(
            device,
            non_blocking=True,
        )

        pred = model(x)

        residual = (
            y - pred
        ).detach().cpu().numpy()[:, 0, :, :]

        mask = (
            unknown_mask.detach()
            .cpu()
            .numpy()[:, 0, :, :]
            > 0.5
        )

        if accumulator is None:
            accumulator = _empty_spatial_accumulator(
                residual.shape[-2:]
            )

        valid_residual = np.where(
            mask,
            residual,
            0.0,
        )
        abs_residual = np.abs(valid_residual)

        accumulator['count'] += np.sum(
            mask,
            axis=0,
            dtype=np.int64,
        )
        accumulator['sum'] += np.sum(
            valid_residual,
            axis=0,
            dtype=np.float64,
        )
        accumulator['sum_sq'] += np.sum(
            valid_residual ** 2,
            axis=0,
            dtype=np.float64,
        )
        accumulator['abs_sum'] += np.sum(
            abs_residual,
            axis=0,
            dtype=np.float64,
        )
        accumulator['abs_sum_sq'] += np.sum(
            abs_residual ** 2,
            axis=0,
            dtype=np.float64,
        )

    if accumulator is None:
        raise ValueError('No batches available for spatial diagnostics.')

    return accumulator


def summarize_spatial_accumulator(
    acc: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    count = acc['count'].astype(np.float64)

    mean = np.full(
        count.shape,
        np.nan,
        dtype=np.float64,
    )
    rmse = np.full_like(mean, np.nan)
    mae = np.full_like(mean, np.nan)
    var = np.full_like(mean, np.nan)
    abs_var = np.full_like(mean, np.nan)

    valid = count > 0

    mean[valid] = (
        acc['sum'][valid]
        / count[valid]
    )
    rmse[valid] = np.sqrt(
        acc['sum_sq'][valid]
        / count[valid]
    )
    mae[valid] = (
        acc['abs_sum'][valid]
        / count[valid]
    )

    valid_var = count > 1

    var_numerator = (
        acc['sum_sq']
        - np.divide(
            acc['sum'] ** 2,
            count,
            out=np.zeros_like(acc['sum']),
            where=count > 0,
        )
    )
    abs_var_numerator = (
        acc['abs_sum_sq']
        - np.divide(
            acc['abs_sum'] ** 2,
            count,
            out=np.zeros_like(acc['abs_sum']),
            where=count > 0,
        )
    )

    var[valid_var] = np.maximum(
        var_numerator[valid_var]
        / (count[valid_var] - 1.0),
        0.0,
    )
    abs_var[valid_var] = np.maximum(
        abs_var_numerator[valid_var]
        / (count[valid_var] - 1.0),
        0.0,
    )

    return {
        'count': acc['count'],
        'mean': mean,
        'rmse': rmse,
        'mae': mae,
        'var': var,
        'std': np.sqrt(var),
        'abs_var': abs_var,
        'abs_std': np.sqrt(abs_var),
    }


def _percentile_abs(
    arr: np.ndarray,
    percentile: float = 98.0,
    fallback: float = 1.0,
) -> float:
    finite = np.abs(
        arr[np.isfinite(arr)]
    )
    if finite.size == 0:
        return fallback

    value = float(
        np.nanpercentile(
            finite,
            percentile,
        )
    )
    if not np.isfinite(value) or value <= 0.0:
        return fallback
    return value


def _percentile_positive(
    arr: np.ndarray,
    percentile: float = 98.0,
    fallback: float = 1.0,
) -> float:
    finite = arr[
        np.isfinite(arr)
        & (arr >= 0.0)
    ]
    if finite.size == 0:
        return fallback

    value = float(
        np.nanpercentile(
            finite,
            percentile,
        )
    )
    if not np.isfinite(value) or value <= 0.0:
        return fallback
    return value


def area_weighted_mean(field: np.ndarray, lat: np.ndarray) -> float:
    weights = np.broadcast_to(
        np.cos(np.deg2rad(lat)).reshape(-1, 1),
        field.shape,
    )
    valid = np.isfinite(field)
    if not np.any(valid):
        return float("nan")
    weighted_sum = np.sum(field[valid] * weights[valid])
    return float(weighted_sum / np.sum(weights[valid]))


def frequency_area_weighted_mean(field, lat, frequency=None):
    area = np.broadcast_to(np.cos(np.deg2rad(lat)).reshape(-1, 1), field.shape)
    freq = np.ones_like(field, dtype=float) if frequency is None else np.asarray(frequency, dtype=float)
    valid = np.isfinite(field) & np.isfinite(freq) & (freq > 0)
    if not np.any(valid):
        return float("nan")
    weights = area * freq
    return float(np.sum(field[valid] * weights[valid]) / np.sum(weights[valid]))


def plot_spatial_diagnostics(output_dir, lat, lon, val_stats, target_stats):
    output_dir.mkdir(parents=True, exist_ok=True)
    val = val_stats["mean"].astype(float)
    target = target_stats["mean"].astype(float)
    difference = target - val
    val_count = val_stats["count"].astype(float)
    target_count = target_stats["count"].astype(float)
    difference_count = np.minimum(val_count, target_count)
    fields = [val, target, difference, target_count]
    frequencies = [val_count, target_count, difference_count, None]
    titles = ["Validation", "Counterfactual", "Counterfactual - Validation", "Counterfactual accu_sox-mask frequency"]
    cmaps = ["RdBu_r", "RdBu_r", "RdBu_r", "viridis"]
    labels = ["Delta lnNd", "Delta lnNd", "Delta lnNd", "Number of days"]
    shared = _percentile_abs(np.concatenate([val[np.isfinite(val)], target[np.isfinite(target)]]), 98.0, 1.0)
    vmaxes = [shared, shared, _percentile_abs(difference, 98.0, 1.0), max(float(np.nanmax(target_count)), 1.0)]
    if HAS_CARTOPY:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), subplot_kw={"projection": ccrs.PlateCarree()}, constrained_layout=True)
    else:
        fig, axes = plt.subplots(2, 2, figsize=(15, 10), constrained_layout=True)
    for ax, field, frequency, title, cmap, label, vmax in zip(axes.flat, fields, frequencies, titles, cmaps, labels, vmaxes):
        kwargs = {"shading": "auto", "cmap": cmap, "vmin": -vmax if cmap == "RdBu_r" else 0.0, "vmax": vmax}
        if HAS_CARTOPY:
            kwargs["transform"] = ccrs.PlateCarree()
            ax.coastlines(resolution="110m", linewidth=0.7)
        else:
            ax.set_xlabel("Longitude")
            ax.set_ylabel("Latitude")
            ax.grid(linewidth=0.3, alpha=0.4)
        mesh = ax.pcolormesh(lon, lat, field, **kwargs)
        mean_value = frequency_area_weighted_mean(field, lat, frequency)
        ax.set_title(f"{title}\nArea- and frequency-weighted mean: {mean_value:.4f}")
        fig.colorbar(mesh, ax=ax, shrink=0.88, pad=0.03, label=label)
    output_path = output_dir.joinpath(f"spatial_diagnostics_{OUTPUT_TAG}.png")
    fig.savefig(output_path, dpi=250, bbox_inches="tight")
    plt.close(fig)
    return [output_path]


# ------------------------------------------------------------
# 数据层：pretrain_random_mask.py 和 unet_finetune_pretrained.py
# 共用同一份实现，保证「输入变量顺序 / 标准化 / 掩膜口径 / 缓存格式」
# 完全一致，两个脚本之间不会互相冲突。
#
# 缓存格式（每个日文件一个 .npy，形状 (n_feature + 2, H, W) float32）：
#     [0:n_feature] = 原始 feature 通道（顺序 = feature_vars）
#     [n_feature]   = 原始目标列（lnNd）
#     [n_feature+1] = 原始 accu_sox（做 SOx 掩膜要用）
# 存原始值，标准化在取数时按当前 feature_stats 现算，
# 所以统计量变化（改训练子集/年份）不需要重建缓存；
# 只有 feature_vars 改变才需要重建。
# ------------------------------------------------------------
def select_feature_vars(sample_path: Path) -> list[str]:
    """
    Fixed input variables for U-Net.

    Only use selected meteorological and MODIS variables.
    Latitude and longitude channels are not used.

    两个脚本必须用同一份定义（通道顺序 = 输入通道顺序）。
    """
    selected_vars = [
        'ssza_mod08',
        'ssaa_mod08',
        'slza_mod08',
        'slaa_mod08',
        'pv_1000',
        'pv_750',
        'pv_500',
        'r_1000',
        'r_750',
        'r_500',
        't_1000',
        't_750',
        't_500',
        'u_1000',
        'u_750',
        'u_500',
        'v_1000',
        'v_750',
        'v_500',
        'u10',
        'v10',
        'd2m',
        't2m',
        'msl',
        'sst',
        'sp',
        'blh',
        'cape',
        'tp',
        'lsp',
        'bld',
        'ptype',
    ]

    with xr.open_dataset(sample_path) as ds:
        missing = [
            name for name in selected_vars
            if name not in ds.data_vars
        ]

    if missing:
        raise ValueError(
            "Missing required feature variables: "
            + ", ".join(missing)
        )

    return selected_vars


def standardize(
    arr: np.ndarray,
    mean: float,
    std: float,
) -> np.ndarray:
    out = (arr.astype(np.float32) - np.float32(mean)) / np.float32(std)
    return np.nan_to_num(
        out,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


def build_raw_stack(
    path: Path,
    feature_vars: list[str],
    target_col: str = 'lnnd',
    sox_var: str = 'accu_sox',
) -> np.ndarray:
    """
    读一个日文件，返回 (n_feature + 2, H, W) 的 float32 原始值数组。
    这就是缓存里存的东西，也是两个脚本共用的取数入口。
    """
    channels = []
    with xr.open_dataset(path) as ds:
        for name in feature_vars:
            channels.append(ds[name].values.astype(np.float32))
        channels.append(ds[target_col].values.astype(np.float32))
        channels.append(ds[sox_var].values.astype(np.float32))

    return np.stack(channels, axis=0).astype(np.float32)


def build_mask_from_accu_sox(
    current_accu_sox: np.ndarray,
    target_valid: np.ndarray,
    start_quantile: float,
    fraction: float,
) -> np.ndarray:
    """
    在 target 有效、且 accu_sox 有效的格点内，按 accu_sox 升序排序，
    取分位点区间 [start_quantile, start_quantile + fraction] 的格点作为掩膜。

    即「当前样本自身 accu_sox 最高的那一档格点」。例如：
        start_quantile=0.90, fraction=0.10 -> 最高 10%
        start_quantile=0.60, fraction=0.40 -> 最高 40%

    start/end 用 floor / ceil 换算并做 clamp，保证至少取到 1 个格点。
    两个脚本都用这个函数，避免出现两套掩膜定义。
    """
    sox = current_accu_sox.astype(np.float64)
    candidate = np.isfinite(sox) & (target_valid > 0.5)

    mask = np.zeros(sox.shape, dtype=np.float32)
    flat_idx = np.flatnonzero(candidate.ravel())
    n_valid = flat_idx.size
    if n_valid == 0:
        return mask

    values = sox.ravel()[flat_idx]
    sorted_local = np.argsort(values)

    start = int(np.floor(start_quantile * n_valid))
    end = int(np.ceil(
        (start_quantile + fraction) * n_valid
    ))
    start = min(max(start, 0), n_valid - 1)
    end = min(max(end, start + 1), n_valid)

    selected_local = sorted_local[start:end]
    mask.ravel()[flat_idx[selected_local]] = 1.0
    return mask


def cache_path_for(cache_dir: Path, path: Path) -> Path:
    return cache_dir / f'{path.stem}.npy'


def cache_meta_path(cache_dir: Path) -> Path:
    return cache_dir / 'cache_meta.json'


def save_cache_metadata(
    cache_dir: Path,
    feature_vars: list[str],
    n_files: int,
) -> None:
    save_json(
        cache_meta_path(cache_dir),
        {
            'feature_vars': list(feature_vars),
            'n_cached_files': int(n_files),
            'cache_layout': 'raw_features + raw_target + raw_accu_sox',
        },
    )


def validate_cache_metadata(
    cache_dir: Path,
    feature_vars: list[str],
) -> None:
    """
    缓存按 feature_vars 的顺序排列通道，变量列表一变旧缓存就不能用；
    这里直接报错要求重建，避免静默用错通道顺序。
    """
    meta_path = cache_meta_path(cache_dir)
    if not meta_path.exists():
        raise FileNotFoundError(
            f'Cache directory {cache_dir} has no cache_meta.json, '
            'cannot verify that its contents match the current '
            'feature_vars. Set REBUILD_CACHE=True '
            'or delete the cache directory.'
        )

    meta = json.loads(meta_path.read_text(encoding='utf-8'))

    if list(meta.get('feature_vars', [])) != list(feature_vars):
        raise ValueError(
            'Cached feature_vars differ from the current ones. '
            'Set REBUILD_CACHE=True or delete the cache directory.'
        )


def prepare_preprocessed_cache(
    paths: list[Path],
    feature_vars: list[str],
    cache_dir: Path,
    rebuild: bool = False,
) -> None:
    """
    把所有需要用到的日文件预处理成 .npy 缓存（缺哪个补哪个）。

    两个脚本共用同一个 cache_dir，所以谁先跑谁建，另一个直接复用。
    """
    cache_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    if rebuild:
        missing = list(paths)
    else:
        if cache_meta_path(cache_dir).exists():
            validate_cache_metadata(cache_dir, feature_vars)
        missing = [
            path
            for path in paths
            if not cache_path_for(cache_dir, path).exists()
        ]

    n_channels = len(feature_vars) + 2
    print(f'Preprocessed cache: {cache_dir}')
    print(
        f'  files: {len(paths) - len(missing)}/{len(paths)} cached, '
        f'{len(missing)} to build | '
        f'{n_channels}x180x360 float32 = '
        f'{n_channels * 180 * 360 * 4 / 1e6:.1f} MB/file, '
        f'total {len(paths) * n_channels * 180 * 360 * 4 / 1e9:.1f} GB'
    )

    if missing:
        print(f'  building cache for {len(missing)} files...')
        total = len(missing)
        for index, path in enumerate(missing, start=1):
            target = cache_path_for(cache_dir, path)
            stack = build_raw_stack(
                path,
                feature_vars,
            )
            # 先写临时文件再原子替换，避免中断时留下半截文件被当成有效缓存。
            tmp_path = target.with_suffix('.tmp.npy')
            np.save(tmp_path, stack)
            os.replace(tmp_path, target)

            if index % 200 == 0 or index == total:
                print(
                    f'    cached {index}/{total} files',
                    flush=True,
                )

    save_cache_metadata(
        cache_dir,
        feature_vars,
        len(paths),
    )
    print('  preprocessed cache ready.')

