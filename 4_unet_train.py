from __future__ import annotations

import datetime as dt
import json
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import torch
import torch.nn as nn
import xarray as xr
from scipy.stats import t as student_t
from torch.utils.data import DataLoader, Dataset

try:
    import cartopy.crs as ccrs
    import cartopy.feature as cfeature

    HAS_CARTOPY = True
except ImportError:
    HAS_CARTOPY = False


DATA_ROOT = Path('/home/chenyiqi/260320_ship_emission/processed_data/ml_grid_data')
OUT_DIR = Path('/home/chenyiqi/260320_ship_emission/processed_data/unet_lnnd')

TRAIN_YEARS = tuple(range(2000, 2018))
VAL_YEARS = (2018, 2019)

# 用这些年份计算 lnNd_observed - lnNd_counterfactual。
# 当前设置为 2020；如果以后要分析 2020–2022，可改成 (2020, 2021, 2022)。
COUNTERFACTUAL_YEARS = (2020,)

# 每个样本中，在 lnNd 有效格点内，按 accu_sox 从大到小精确选最高 10%。
MASK_QUANTILE = 0.90

TARGET_COL = 'lnnd'
RANDOM_STATE = 42
BATCH_SIZE = 2
NUM_EPOCHS = 20
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
KNOWN_LOSS_WEIGHT = 0.05
NUM_WORKERS = 2

# ------------------------------------------------------------
# 模型使用方式：
# True  -> 直接加载已有 best_unet_lnnd.pt，跳过训练；
# False -> 从头重新训练，最多训练 NUM_EPOCHS=20 个 epoch。
# ------------------------------------------------------------
USE_SAVED_MODEL = True
SAVED_MODEL_PATH = Path(
    '/home/chenyiqi/260320_ship_emission/processed_data/unet_lnnd/best_unet_lnnd.pt'
)

# 逐格点显著性比较至少需要的验证期/目标期样本数。
MIN_GRID_SAMPLES = 10
SIGNIFICANCE_LEVEL = 0.05

# Same spirit as the old RF feature selection: keep meteorology and geometry,
# but do not feed cloud target variables or SOx variables directly.
EXCLUDE_VARS = {
    'accu_sox',
    'source_sox',
    'valid_accu_sox',
    'weighted_sox_diff',
    'cf_ret_liq_mod08',
    'cot_mod08',
    'cer_mod08',
    'cwp_mod08',
    'lnnd',
    'nd',
    'cf_ret_combined_mod08',
    'aod_mod08',
}


def seed_everything(seed: int) -> None:
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)


def parse_date_from_path(path: Path) -> dt.date | None:
    match = re.search(r'_(\d{8})\d{4}\.nc$', path.name)
    if not match:
        return None
    return dt.datetime.strptime(match.group(1), '%Y%m%d').date()


def collect_grid_files(data_root: Path, years: tuple[int, ...]) -> list[Path]:
    paths = []
    for path in sorted(data_root.glob('*/ml_grid_*.nc')):
        date_value = parse_date_from_path(path)
        if date_value is not None and date_value.year in years:
            paths.append(path)
    if not paths:
        raise ValueError(f'No daily grid files found in {data_root} for years={years}')
    return paths


def select_feature_vars(sample_path: Path) -> list[str]:
    with xr.open_dataset(sample_path) as ds:
        feature_vars = []
        for name, da in ds.data_vars.items():
            if name in EXCLUDE_VARS:
                continue
            if da.ndim != 2 or da.dims != ('lat', 'lon'):
                continue
            if not np.issubdtype(da.dtype, np.number):
                continue
            if name.endswith('_0'):
                continue
            feature_vars.append(name)
    if not feature_vars:
        raise ValueError(f'No feature variables found in {sample_path}')
    return feature_vars


def build_mask_from_accu_sox(
    current_accu_sox: np.ndarray,
    target_valid: np.ndarray,
) -> np.ndarray:
    """
    在 target 有效的格点内，按照当前样本自身的 accu_sox 从大到小排序，
    精确选取最高 (1 - MASK_QUANTILE) = 10% 作为 unknown 区域。

    与旧代码不同：
        旧：historical accu_sox - 2020 accu_sox 最大的 10%
        新：当前样本 accu_sox 本身最大的 10%
    """
    sox = current_accu_sox.astype(np.float64)
    candidate = np.isfinite(sox) & (target_valid > 0.5)

    mask = np.zeros(sox.shape, dtype=np.float32)
    flat_idx = np.flatnonzero(candidate.ravel())
    n_valid = flat_idx.size
    if n_valid == 0:
        return mask

    fraction = 1.0 - MASK_QUANTILE
    n_mask = max(1, int(np.ceil(fraction * n_valid)))
    n_mask = min(n_mask, n_valid)

    values = sox.ravel()[flat_idx]

    # argpartition 比完整排序更快；得到 SOx 最大的 n_mask 个格点。
    top_local = np.argpartition(values, n_valid - n_mask)[n_valid - n_mask:]
    top_flat_idx = flat_idx[top_local]

    mask.ravel()[top_flat_idx] = 1.0
    return mask


def add_lat_lon_channels(ds: xr.Dataset) -> tuple[np.ndarray, np.ndarray]:
    lat = ds['lat'].values.astype(np.float32)
    lon = ds['lon'].values.astype(np.float32)
    lat_grid = np.repeat(lat[:, None], lon.size, axis=1)
    lon_grid = np.repeat(lon[None, :], lat.size, axis=0)
    return lat_grid / 90.0, lon_grid / 180.0


def get_grid_coordinates(sample_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(sample_path) as ds:
        lat = ds['lat'].values.astype(np.float64)
        lon = ds['lon'].values.astype(np.float64)
    return lat, lon


def compute_feature_stats(
    paths: list[Path],
    feature_vars: list[str],
) -> dict[str, tuple[float, float]]:
    sums = {name: 0.0 for name in feature_vars}
    sums_sq = {name: 0.0 for name in feature_vars}
    counts = {name: 0 for name in feature_vars}

    for path in paths:
        with xr.open_dataset(path) as ds:
            for name in feature_vars:
                arr = ds[name].values.astype(np.float64)
                valid = np.isfinite(arr)
                if not np.any(valid):
                    continue
                vals = arr[valid]
                sums[name] += float(np.sum(vals))
                sums_sq[name] += float(np.sum(vals * vals))
                counts[name] += int(vals.size)

    stats = {}
    for name in feature_vars:
        if counts[name] == 0:
            stats[name] = (0.0, 1.0)
            continue
        mean = sums[name] / counts[name]
        var = max(sums_sq[name] / counts[name] - mean * mean, 1e-12)
        stats[name] = (float(mean), float(np.sqrt(var)))
    return stats


def standardize(arr: np.ndarray, mean: float, std: float) -> np.ndarray:
    out = (arr.astype(np.float32) - np.float32(mean)) / np.float32(std)
    return np.nan_to_num(
        out,
        nan=0.0,
        posinf=0.0,
        neginf=0.0,
    ).astype(np.float32)


class LnndGridDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        feature_vars: list[str],
        feature_stats: dict[str, tuple[float, float]],
    ):
        self.paths = paths
        self.feature_vars = feature_vars
        self.feature_stats = feature_stats

    def __len__(self) -> int:
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = self.paths[index]

        with xr.open_dataset(path) as ds:
            lnnd_true = ds[TARGET_COL].values.astype(np.float32)
            current_accu_sox = ds['accu_sox'].values.astype(np.float32)

            target_valid = np.isfinite(lnnd_true).astype(np.float32)

            # 新定义：当前样本 accu_sox 最大的 10%。
            unknown_mask = build_mask_from_accu_sox(
                current_accu_sox=current_accu_sox,
                target_valid=target_valid,
            )
            known_mask = (1.0 - unknown_mask) * target_valid

            # unknown 区域的 lnNd 不给模型看，用当前样本 known 区域均值占位。
            known_values = lnnd_true[known_mask > 0.5]
            if known_values.size:
                known_mean = float(np.nanmean(known_values))
            else:
                known_mean = float(np.nanmean(lnnd_true))

            if not np.isfinite(known_mean):
                known_mean = 0.0

            lnnd_known = lnnd_true.copy()
            lnnd_known[unknown_mask > 0.5] = known_mean
            lnnd_known = np.nan_to_num(
                lnnd_known,
                nan=known_mean,
                posinf=known_mean,
                neginf=known_mean,
            )

            channels = []
            for name in self.feature_vars:
                mean, std = self.feature_stats[name]
                channels.append(
                    standardize(ds[name].values, mean, std)
                )

            lat_channel, lon_channel = add_lat_lon_channels(ds)
            channels.extend([
                lat_channel.astype(np.float32),
                lon_channel.astype(np.float32),
            ])
            channels.append(unknown_mask.astype(np.float32))
            channels.append(lnnd_known.astype(np.float32))

        x = np.stack(channels, axis=0).astype(np.float32)

        y = np.nan_to_num(
            lnnd_true,
            nan=known_mean,
            posinf=known_mean,
            neginf=known_mean,
        )[None, ...].astype(np.float32)

        return {
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y),
            'unknown_mask': torch.from_numpy(
                unknown_mask[None, ...].astype(np.float32)
            ),
            'known_mask': torch.from_numpy(
                known_mask[None, ...].astype(np.float32)
            ),
        }


class DoubleConv(nn.Module):
    def __init__(self, in_channels: int, out_channels: int):
        super().__init__()
        self.net = nn.Sequential(
            nn.Conv2d(
                in_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
            nn.Conv2d(
                out_channels,
                out_channels,
                kernel_size=3,
                padding=1,
                bias=False,
            ),
            nn.BatchNorm2d(out_channels),
            nn.ReLU(inplace=True),
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        return self.net(x)


class UNet(nn.Module):
    def __init__(
        self,
        in_channels: int,
        out_channels: int = 1,
        widths: tuple[int, ...] = (32, 64, 128, 256),
    ):
        super().__init__()
        self.downs = nn.ModuleList()
        self.pools = nn.ModuleList()

        current = in_channels
        for width in widths:
            self.downs.append(DoubleConv(current, width))
            self.pools.append(nn.MaxPool2d(2))
            current = width

        self.bottleneck = DoubleConv(
            widths[-1],
            widths[-1] * 2,
        )

        self.up_transpose = nn.ModuleList()
        self.ups = nn.ModuleList()

        current = widths[-1] * 2
        for width in reversed(widths):
            self.up_transpose.append(
                nn.ConvTranspose2d(
                    current,
                    width,
                    kernel_size=2,
                    stride=2,
                )
            )
            self.ups.append(
                DoubleConv(width * 2, width)
            )
            current = width

        self.head = nn.Conv2d(
            widths[0],
            out_channels,
            kernel_size=1,
        )

    def forward(self, x: torch.Tensor) -> torch.Tensor:
        skips = []

        for down, pool in zip(self.downs, self.pools):
            x = down(x)
            skips.append(x)
            x = pool(x)

        x = self.bottleneck(x)

        for up_t, up, skip in zip(
            self.up_transpose,
            self.ups,
            reversed(skips),
        ):
            x = up_t(x)

            if x.shape[-2:] != skip.shape[-2:]:
                x = nn.functional.interpolate(
                    x,
                    size=skip.shape[-2:],
                    mode='bilinear',
                    align_corners=False,
                )

            x = torch.cat([skip, x], dim=1)
            x = up(x)

        return self.head(x)


def masked_mse(
    pred: torch.Tensor,
    target: torch.Tensor,
    mask: torch.Tensor,
) -> torch.Tensor:
    denom = torch.clamp(mask.sum(), min=1.0)
    return torch.sum(
        ((pred - target) ** 2) * mask
    ) / denom


def train_one_epoch(
    model: nn.Module,
    loader: DataLoader,
    optimizer: torch.optim.Optimizer,
    device: torch.device,
) -> float:
    model.train()
    losses = []

    for batch in loader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        unknown_mask = batch['unknown_mask'].to(device)
        known_mask = batch['known_mask'].to(device)

        pred = model(x)

        loss_unknown = masked_mse(
            pred,
            y,
            unknown_mask,
        )
        loss_known = masked_mse(
            pred,
            y,
            known_mask,
        )

        loss = (
            loss_unknown
            + KNOWN_LOSS_WEIGHT * loss_known
        )

        optimizer.zero_grad(set_to_none=True)
        loss.backward()
        optimizer.step()

        losses.append(
            float(loss.detach().cpu())
        )

    return (
        float(np.mean(losses))
        if losses
        else float('nan')
    )


@torch.no_grad()
def evaluate(
    model: nn.Module,
    loader: DataLoader,
    device: torch.device,
) -> dict[str, float]:
    model.eval()

    total_unknown_loss = 0.0
    total_known_loss = 0.0
    total_unknown_count = 0.0
    total_known_count = 0.0

    for batch in loader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        unknown_mask = batch['unknown_mask'].to(device)
        known_mask = batch['known_mask'].to(device)

        pred = model(x)

        total_unknown_loss += float(
            torch.sum(
                ((pred - y) ** 2) * unknown_mask
            ).cpu()
        )
        total_known_loss += float(
            torch.sum(
                ((pred - y) ** 2) * known_mask
            ).cpu()
        )

        total_unknown_count += float(
            unknown_mask.sum().cpu()
        )
        total_known_count += float(
            known_mask.sum().cpu()
        )

    unknown_mse = (
        total_unknown_loss
        / max(total_unknown_count, 1.0)
    )
    known_mse = (
        total_known_loss
        / max(total_known_count, 1.0)
    )

    return {
        'unknown_rmse': float(np.sqrt(unknown_mse)),
        'known_rmse': float(np.sqrt(known_mse)),
        'unknown_mse': float(unknown_mse),
        'known_mse': float(known_mse),
    }


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


def global_stats_from_accumulator(
    acc: dict[str, np.ndarray],
) -> dict[str, float]:
    n = int(np.sum(acc['count']))
    if n == 0:
        return {
            'count': 0,
            'mean': np.nan,
            'rmse': np.nan,
            'mae': np.nan,
            'std': np.nan,
            'abs_std': np.nan,
        }

    total_sum = float(np.sum(acc['sum']))
    total_sum_sq = float(np.sum(acc['sum_sq']))
    total_abs_sum = float(np.sum(acc['abs_sum']))
    total_abs_sum_sq = float(
        np.sum(acc['abs_sum_sq'])
    )

    mean = total_sum / n
    rmse = np.sqrt(total_sum_sq / n)
    mae = total_abs_sum / n

    if n > 1:
        var = max(
            (
                total_sum_sq
                - total_sum ** 2 / n
            )
            / (n - 1),
            0.0,
        )
        abs_var = max(
            (
                total_abs_sum_sq
                - total_abs_sum ** 2 / n
            )
            / (n - 1),
            0.0,
        )
    else:
        var = np.nan
        abs_var = np.nan

    return {
        'count': n,
        'mean': float(mean),
        'rmse': float(rmse),
        'mae': float(mae),
        'std': float(np.sqrt(var)),
        'abs_std': float(np.sqrt(abs_var)),
    }


def welch_compare_target_abs_vs_validation_abs(
    val_stats: dict[str, np.ndarray],
    target_stats: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    逐格点检验：

        H0: E(|target delta|) <= E(|validation error|)
        H1: E(|target delta|) >  E(|validation error|)

    其中：
        validation error = lnNd_observed - lnNd_predicted
        target delta      = lnNd_observed - lnNd_counterfactual

    使用 Welch t test 的逐格点近似，允许两时期方差不同。
    """
    n_val = val_stats['count'].astype(np.float64)
    n_target = target_stats['count'].astype(np.float64)

    mean_val = val_stats['mae']
    mean_target = target_stats['mae']

    var_val = val_stats['abs_var']
    var_target = target_stats['abs_var']

    t_stat = np.full(
        n_val.shape,
        np.nan,
        dtype=np.float64,
    )
    dof = np.full_like(t_stat, np.nan)
    p_one_sided = np.full_like(t_stat, np.nan)

    valid = (
        (n_val >= MIN_GRID_SAMPLES)
        & (n_target >= MIN_GRID_SAMPLES)
        & np.isfinite(mean_val)
        & np.isfinite(mean_target)
        & np.isfinite(var_val)
        & np.isfinite(var_target)
    )

    term_val = np.zeros_like(t_stat)
    term_target = np.zeros_like(t_stat)

    term_val[valid] = (
        var_val[valid]
        / n_val[valid]
    )
    term_target[valid] = (
        var_target[valid]
        / n_target[valid]
    )

    se2 = term_val + term_target
    valid_se = valid & (se2 > 0.0)

    t_stat[valid_se] = (
        mean_target[valid_se]
        - mean_val[valid_se]
    ) / np.sqrt(se2[valid_se])

    denom = np.zeros_like(t_stat)
    denom[valid_se] = (
        (term_val[valid_se] ** 2)
        / (n_val[valid_se] - 1.0)
        + (term_target[valid_se] ** 2)
        / (n_target[valid_se] - 1.0)
    )

    valid_df = valid_se & (denom > 0.0)

    dof[valid_df] = (
        se2[valid_df] ** 2
        / denom[valid_df]
    )

    p_one_sided[valid_df] = student_t.sf(
        t_stat[valid_df],
        dof[valid_df],
    )

    significant_larger = (
        valid_df
        & (mean_target > mean_val)
        & (p_one_sided < SIGNIFICANCE_LEVEL)
    )

    return {
        't_stat': t_stat,
        'dof': dof,
        'p_one_sided': p_one_sided,
        'significant_larger': significant_larger,
    }


def welch_compare_signed_means(
    val_stats: dict[str, np.ndarray],
    target_stats: dict[str, np.ndarray],
) -> dict[str, np.ndarray]:
    """
    额外给出 signed residual 的两侧 Welch 检验：

        validation residual = observed - predicted
        target delta        = observed - counterfactual

    检验两者均值是否不同。
    """
    n_val = val_stats['count'].astype(np.float64)
    n_target = target_stats['count'].astype(np.float64)

    mean_val = val_stats['mean']
    mean_target = target_stats['mean']

    var_val = val_stats['var']
    var_target = target_stats['var']

    t_stat = np.full(
        n_val.shape,
        np.nan,
        dtype=np.float64,
    )
    dof = np.full_like(t_stat, np.nan)
    p_two_sided = np.full_like(t_stat, np.nan)

    valid = (
        (n_val >= MIN_GRID_SAMPLES)
        & (n_target >= MIN_GRID_SAMPLES)
        & np.isfinite(mean_val)
        & np.isfinite(mean_target)
        & np.isfinite(var_val)
        & np.isfinite(var_target)
    )

    term_val = np.zeros_like(t_stat)
    term_target = np.zeros_like(t_stat)

    term_val[valid] = (
        var_val[valid]
        / n_val[valid]
    )
    term_target[valid] = (
        var_target[valid]
        / n_target[valid]
    )

    se2 = term_val + term_target
    valid_se = valid & (se2 > 0.0)

    t_stat[valid_se] = (
        mean_target[valid_se]
        - mean_val[valid_se]
    ) / np.sqrt(se2[valid_se])

    denom = np.zeros_like(t_stat)
    denom[valid_se] = (
        (term_val[valid_se] ** 2)
        / (n_val[valid_se] - 1.0)
        + (term_target[valid_se] ** 2)
        / (n_target[valid_se] - 1.0)
    )

    valid_df = valid_se & (denom > 0.0)

    dof[valid_df] = (
        se2[valid_df] ** 2
        / denom[valid_df]
    )

    p_two_sided[valid_df] = (
        2.0
        * student_t.sf(
            np.abs(t_stat[valid_df]),
            dof[valid_df],
        )
    )

    return {
        't_stat': t_stat,
        'dof': dof,
        'p_two_sided': p_two_sided,
        'significant_difference': (
            valid_df
            & (
                p_two_sided
                < SIGNIFICANCE_LEVEL
            )
        ),
    }


def _safe_ratio(
    numerator: np.ndarray,
    denominator: np.ndarray,
) -> np.ndarray:
    out = np.full(
        numerator.shape,
        np.nan,
        dtype=np.float64,
    )
    valid = (
        np.isfinite(numerator)
        & np.isfinite(denominator)
        & (denominator > 0.0)
    )
    out[valid] = (
        numerator[valid]
        / denominator[valid]
    )
    return out


def save_spatial_diagnostics(
    output_path: Path,
    lat: np.ndarray,
    lon: np.ndarray,
    val_stats: dict[str, np.ndarray],
    target_stats: dict[str, np.ndarray],
    abs_test: dict[str, np.ndarray],
    signed_test: dict[str, np.ndarray],
) -> None:
    ratio_mae = _safe_ratio(
        target_stats['mae'],
        val_stats['mae'],
    )
    ratio_rmse = _safe_ratio(
        np.abs(target_stats['mean']),
        val_stats['rmse'],
    )

    corrected_signed_effect = (
        target_stats['mean']
        - val_stats['mean']
    )

    ds = xr.Dataset(
        data_vars={
            'val_count': (
                ('lat', 'lon'),
                val_stats['count'].astype(np.int32),
            ),
            'val_error_mean': (
                ('lat', 'lon'),
                val_stats['mean'].astype(np.float32),
            ),
            'val_error_mae': (
                ('lat', 'lon'),
                val_stats['mae'].astype(np.float32),
            ),
            'val_error_rmse': (
                ('lat', 'lon'),
                val_stats['rmse'].astype(np.float32),
            ),
            'val_error_std': (
                ('lat', 'lon'),
                val_stats['std'].astype(np.float32),
            ),
            'target_count': (
                ('lat', 'lon'),
                target_stats['count'].astype(np.int32),
            ),
            'delta_lnnd_mean': (
                ('lat', 'lon'),
                target_stats['mean'].astype(np.float32),
            ),
            'delta_lnnd_abs_mean': (
                ('lat', 'lon'),
                target_stats['mae'].astype(np.float32),
            ),
            'delta_lnnd_std': (
                ('lat', 'lon'),
                target_stats['std'].astype(np.float32),
            ),
            'target_abs_to_val_mae_ratio': (
                ('lat', 'lon'),
                ratio_mae.astype(np.float32),
            ),
            'abs_mean_delta_to_val_rmse_ratio': (
                ('lat', 'lon'),
                ratio_rmse.astype(np.float32),
            ),
            'bias_corrected_delta_lnnd': (
                ('lat', 'lon'),
                corrected_signed_effect.astype(np.float32),
            ),
            'abs_welch_t': (
                ('lat', 'lon'),
                abs_test['t_stat'].astype(np.float32),
            ),
            'abs_welch_p_one_sided': (
                ('lat', 'lon'),
                abs_test['p_one_sided'].astype(np.float32),
            ),
            'abs_target_significantly_larger': (
                ('lat', 'lon'),
                abs_test['significant_larger'].astype(np.int8),
            ),
            'signed_welch_t': (
                ('lat', 'lon'),
                signed_test['t_stat'].astype(np.float32),
            ),
            'signed_welch_p_two_sided': (
                ('lat', 'lon'),
                signed_test['p_two_sided'].astype(np.float32),
            ),
            'signed_target_diff_significant': (
                ('lat', 'lon'),
                signed_test[
                    'significant_difference'
                ].astype(np.int8),
            ),
        },
        coords={
            'lat': lat,
            'lon': lon,
        },
        attrs={
            'mask_definition': (
                f'Top {100 * (1 - MASK_QUANTILE):.1f}% '
                'of current-sample accu_sox among valid lnNd grid cells.'
            ),
            'validation_years': ','.join(
                str(y) for y in VAL_YEARS
            ),
            'counterfactual_years': ','.join(
                str(y)
                for y in COUNTERFACTUAL_YEARS
            ),
            'residual_definition': (
                'observed_minus_reconstructed'
            ),
            'delta_definition': (
                'lnNd_observed_minus_lnNd_counterfactual'
            ),
            'abs_test_definition': (
                'One-sided Welch test of '
                'mean(|target delta|) > '
                'mean(|validation error|).'
            ),
            'signed_test_definition': (
                'Two-sided Welch test of '
                'mean(target signed delta) != '
                'mean(validation signed residual).'
            ),
            'alpha': SIGNIFICANCE_LEVEL,
            'min_grid_samples': MIN_GRID_SAMPLES,
        },
    )

    ds.to_netcdf(output_path)


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


def _plot_single_spatial_field(
    output_path: Path,
    lat: np.ndarray,
    lon: np.ndarray,
    field: np.ndarray,
    title: str,
    vmin: float,
    vmax: float,
) -> None:
    """
    绘制一张空间分布图。
    所有 lnNd 差值图均使用以 0 为中心的 RdBu_r 色标。
    """
    if HAS_CARTOPY:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(8.5, 5.5),
            subplot_kw={
                'projection': ccrs.PlateCarree()
            },
            constrained_layout=True,
        )
        transform = ccrs.PlateCarree()
    else:
        fig, ax = plt.subplots(
            1,
            1,
            figsize=(8.5, 5.5),
            constrained_layout=True,
        )
        transform = None

    kwargs = {
        'shading': 'auto',
        'cmap': 'RdBu_r',
        'vmin': vmin,
        'vmax': vmax,
    }
    if transform is not None:
        kwargs['transform'] = transform

    mesh = ax.pcolormesh(
        lon,
        lat,
        field,
        **kwargs,
    )

    if HAS_CARTOPY:
        ax.coastlines(
            resolution='110m',
            linewidth=0.7,
        )
        ax.add_feature(
            cfeature.BORDERS,
            linewidth=0.35,
        )
        gl = ax.gridlines(
            draw_labels=True,
            linewidth=0.3,
            alpha=0.5,
        )
        gl.top_labels = False
        gl.right_labels = False
    else:
        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.grid(
            linewidth=0.3,
            alpha=0.4,
        )

    ax.set_title(
        title,
        fontsize=13,
    )

    cbar = fig.colorbar(
        mesh,
        ax=ax,
        shrink=0.90,
        pad=0.03,
    )
    cbar.set_label(
        r'$\Delta \ln N_d$'
    )

    fig.savefig(
        output_path,
        dpi=250,
        bbox_inches='tight',
    )
    plt.close(fig)


def plot_spatial_diagnostics(
    output_dir: Path,
    lat: np.ndarray,
    lon: np.ndarray,
    val_stats: dict[str, np.ndarray],
    target_stats: dict[str, np.ndarray],
) -> list[Path]:
    """
    输出三张独立空间图：

    1. VAL_YEARS:
       mean(lnNd_obs - lnNd_predict)

    2. 2020:
       mean(lnNd_obs - lnNd_predict)
       这里的 predict 即 U-Net counterfactual reconstruction。

    3. 2020 - VAL:
       [2020 mean(lnNd_obs - lnNd_predict)]
       -
       [VAL mean(lnNd_obs - lnNd_predict)]

       该量等价于扣除验证期平均系统残差后的 2020 差值。
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    val_residual = (
        val_stats['mean']
        .astype(np.float64)
    )
    target_residual = (
        target_stats['mean']
        .astype(np.float64)
    )

    target_minus_val = (
        target_residual
        - val_residual
    )

    # 第一、第二张图使用完全相同的色标范围，
    # 便于直接比较 VAL 与 2020 的残差幅度和符号。
    shared_vmax = _percentile_abs(
        np.concatenate([
            val_residual[np.isfinite(val_residual)],
            target_residual[np.isfinite(target_residual)],
        ]),
        percentile=98.0,
        fallback=1.0,
    )

    # 第三张图单独使用对称色标。
    difference_vmax = _percentile_abs(
        target_minus_val,
        percentile=98.0,
        fallback=1.0,
    )

    val_year_text = (
        '-'.join(
            str(y)
            for y in VAL_YEARS
        )
    )

    target_year_text = (
        ', '.join(
            str(y)
            for y in COUNTERFACTUAL_YEARS
        )
    )

    path_val = (
        output_dir
        / '01_VALID_lnNd_obs_minus_predict.png'
    )
    path_target = (
        output_dir
        / '02_2020_lnNd_obs_minus_predict.png'
    )
    path_difference = (
        output_dir
        / '03_2020_minus_VALID_residual.png'
    )

    _plot_single_spatial_field(
        output_path=path_val,
        lat=lat,
        lon=lon,
        field=val_residual,
        title=(
            f'Validation ({val_year_text}): '
            r'$\ln N_d^{obs}-\ln N_d^{predict}$'
        ),
        vmin=-shared_vmax,
        vmax=shared_vmax,
    )

    _plot_single_spatial_field(
        output_path=path_target,
        lat=lat,
        lon=lon,
        field=target_residual,
        title=(
            f'{target_year_text}: '
            r'$\ln N_d^{obs}-\ln N_d^{predict}$'
        ),
        vmin=-shared_vmax,
        vmax=shared_vmax,
    )

    _plot_single_spatial_field(
        output_path=path_difference,
        lat=lat,
        lon=lon,
        field=target_minus_val,
        title=(
            f'{target_year_text} residual '
            f'- validation ({val_year_text}) residual'
        ),
        vmin=-difference_vmax,
        vmax=difference_vmax,
    )

    return [
        path_val,
        path_target,
        path_difference,
    ]


def save_global_summary(
    output_path: Path,
    val_acc: dict[str, np.ndarray],
    target_acc: dict[str, np.ndarray],
) -> None:
    val_global = global_stats_from_accumulator(
        val_acc
    )
    target_global = global_stats_from_accumulator(
        target_acc
    )

    # 全部 high-SOx 像元 pooled 后，对 |target delta|
    # 是否大于 |validation error| 做 Welch 检验。
    n_val = val_global['count']
    n_target = target_global['count']

    global_result = {
        'validation': val_global,
        'counterfactual_period': target_global,
    }

    if (
        n_val > 1
        and n_target > 1
        and np.isfinite(val_global['abs_std'])
        and np.isfinite(
            target_global['abs_std']
        )
    ):
        val_abs_var = (
            val_global['abs_std'] ** 2
        )
        target_abs_var = (
            target_global['abs_std'] ** 2
        )

        term_val = val_abs_var / n_val
        term_target = (
            target_abs_var / n_target
        )
        se2 = term_val + term_target

        if se2 > 0.0:
            t_stat = (
                target_global['mae']
                - val_global['mae']
            ) / np.sqrt(se2)

            denom = (
                term_val ** 2 / (n_val - 1)
                + term_target ** 2
                / (n_target - 1)
            )

            if denom > 0.0:
                dof = se2 ** 2 / denom
                p = float(
                    student_t.sf(
                        t_stat,
                        dof,
                    )
                )
            else:
                dof = np.nan
                p = np.nan
        else:
            t_stat = np.nan
            dof = np.nan
            p = np.nan

        global_result[
            'abs_target_vs_validation_test'
        ] = {
            'definition': (
                'One-sided Welch test: '
                'mean(|target delta|) > '
                'mean(|validation error|)'
            ),
            't_stat': float(t_stat),
            'dof': float(dof),
            'p_one_sided': float(p),
            'significant_at_alpha': bool(
                np.isfinite(p)
                and p < SIGNIFICANCE_LEVEL
                and target_global['mae']
                > val_global['mae']
            ),
            'alpha': SIGNIFICANCE_LEVEL,
            'target_abs_to_validation_mae_ratio': float(
                target_global['mae']
                / val_global['mae']
            )
            if val_global['mae'] > 0
            else np.nan,
        }

    output_path.write_text(
        json.dumps(
            global_result,
            indent=2,
        ),
        encoding='utf-8',
    )


def main() -> None:
    seed_everything(RANDOM_STATE)
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    train_paths = collect_grid_files(
        DATA_ROOT,
        TRAIN_YEARS,
    )
    val_paths = collect_grid_files(
        DATA_ROOT,
        VAL_YEARS,
    )
    target_paths = collect_grid_files(
        DATA_ROOT,
        COUNTERFACTUAL_YEARS,
    )

    device = torch.device(
        'cuda'
        if torch.cuda.is_available()
        else 'cpu'
    )

    # ---------------------------------------------------------
    # 特征变量与标准化参数
    #
    # 若使用已保存模型，必须优先使用 checkpoint 中保存的
    # feature_vars / feature_stats，确保输入变量、顺序和标准化方式
    # 与模型训练时完全一致。
    # ---------------------------------------------------------
    checkpoint = None

    if USE_SAVED_MODEL:
        if not SAVED_MODEL_PATH.exists():
            raise FileNotFoundError(
                'USE_SAVED_MODEL=True, but saved model does not exist:\n'
                f'{SAVED_MODEL_PATH}'
            )

        print('=' * 80)
        print('USE_SAVED_MODEL = True')
        print('Skip training and load existing model:')
        print(SAVED_MODEL_PATH)
        print('=' * 80)

        checkpoint = torch.load(
            SAVED_MODEL_PATH,
            map_location='cpu',
            weights_only=False,
        )

        saved_config = checkpoint.get(
            'config',
            {},
        )

        feature_vars = saved_config.get(
            'feature_vars'
        )
        feature_stats_raw = saved_config.get(
            'feature_stats'
        )

        if not feature_vars:
            raise ValueError(
                'The saved checkpoint does not contain config["feature_vars"]. '
                'Cannot safely reconstruct the model input.'
            )

        if not feature_stats_raw:
            raise ValueError(
                'The saved checkpoint does not contain config["feature_stats"]. '
                'Cannot safely reproduce training-time standardization.'
            )

        feature_vars = list(feature_vars)

        missing_stats = [
            name
            for name in feature_vars
            if name not in feature_stats_raw
        ]
        if missing_stats:
            raise ValueError(
                'Saved checkpoint is missing feature statistics for: '
                + ', '.join(missing_stats)
            )

        feature_stats = {
            name: tuple(
                float(v)
                for v in feature_stats_raw[name]
            )
            for name in feature_vars
        }

    else:
        print('=' * 80)
        print('USE_SAVED_MODEL = False')
        print(
            f'Retrain U-Net from scratch for '
            f'{NUM_EPOCHS} epochs.'
        )
        print('=' * 80)

        feature_vars = select_feature_vars(
            train_paths[0]
        )
        feature_stats = compute_feature_stats(
            train_paths,
            feature_vars,
        )

    # ---------------------------------------------------------
    # Dataset / DataLoader
    # ---------------------------------------------------------
    train_ds = LnndGridDataset(
        train_paths,
        feature_vars,
        feature_stats,
    )
    val_ds = LnndGridDataset(
        val_paths,
        feature_vars,
        feature_stats,
    )
    target_ds = LnndGridDataset(
        target_paths,
        feature_vars,
        feature_stats,
    )

    pin_memory = torch.cuda.is_available()

    # 只有重新训练时才真正需要 train_loader。
    train_loader = None
    if not USE_SAVED_MODEL:
        train_loader = DataLoader(
            train_ds,
            batch_size=BATCH_SIZE,
            shuffle=True,
            num_workers=NUM_WORKERS,
            pin_memory=pin_memory,
        )

    val_loader = DataLoader(
        val_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )
    target_loader = DataLoader(
        target_ds,
        batch_size=BATCH_SIZE,
        shuffle=False,
        num_workers=NUM_WORKERS,
        pin_memory=pin_memory,
    )

    in_channels = (
        len(feature_vars)
        + 4
    )  # selected features + lat + lon + M + known lnNd

    # 如果 checkpoint 里记录了输入通道数，检查它是否与当前重建结果一致。
    if USE_SAVED_MODEL:
        saved_input_channels = checkpoint.get(
            'config',
            {},
        ).get(
            'input_channels'
        )

        if (
            saved_input_channels is not None
            and int(saved_input_channels)
            != in_channels
        ):
            raise ValueError(
                'Saved checkpoint input-channel mismatch: '
                f'checkpoint={saved_input_channels}, '
                f'current={in_channels}.'
            )

    model = UNet(
        in_channels=in_channels
    ).to(device)

    config = {
        'train_years': TRAIN_YEARS,
        'val_years': VAL_YEARS,
        'counterfactual_years': COUNTERFACTUAL_YEARS,
        'mask_quantile': MASK_QUANTILE,
        'mask_definition': (
            f'top {100 * (1 - MASK_QUANTILE):.1f}% '
            'of current-sample accu_sox among valid lnNd grid cells'
        ),
        'feature_vars': feature_vars,
        'extra_channels': [
            'lat',
            'lon',
            'M_unknown',
            'lnnd_known_fill',
        ],
        'feature_stats': feature_stats,
        'target': TARGET_COL,
        'input_channels': in_channels,
        'min_grid_samples': MIN_GRID_SAMPLES,
        'significance_level': SIGNIFICANCE_LEVEL,
        'use_saved_model': USE_SAVED_MODEL,
        'saved_model_path': (
            str(SAVED_MODEL_PATH)
            if USE_SAVED_MODEL
            else None
        ),
        'num_epochs_if_retraining': NUM_EPOCHS,
    }

    (
        OUT_DIR
        / 'config.json'
    ).write_text(
        json.dumps(
            config,
            indent=2,
        ),
        encoding='utf-8',
    )

    print(f'Device: {device}')
    print(
        f'Train files: {len(train_paths)}, '
        f'val files: {len(val_paths)}, '
        f'target files: {len(target_paths)}'
    )
    print(
        f'Input channels: {in_channels}'
    )
    print(
        'Feature vars: '
        + ', '.join(feature_vars)
    )
    print(
        'Mask: top '
        f'{100 * (1 - MASK_QUANTILE):.1f}% '
        'of accu_sox in each sample'
    )

    # ---------------------------------------------------------
    # A. 直接使用已经保存的模型
    # ---------------------------------------------------------
    if USE_SAVED_MODEL:
        model.load_state_dict(
            checkpoint['model_state_dict']
        )
        model.eval()

        print('')
        print('Loaded saved model successfully.')

        if 'epoch' in checkpoint:
            print(
                'Saved-model epoch: '
                f'{checkpoint["epoch"]}'
            )

        saved_metrics = checkpoint.get(
            'metrics',
            {},
        )
        if 'unknown_rmse' in saved_metrics:
            print(
                'Saved validation unknown RMSE: '
                f'{saved_metrics["unknown_rmse"]:.6f}'
            )

    # ---------------------------------------------------------
    # B. 从头重新训练，固定训练 NUM_EPOCHS=20 个 epoch
    # ---------------------------------------------------------
    else:
        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        best_unknown_rmse = float('inf')
        history = []

        # 新训练得到的最佳模型仍保存在原位置。
        best_model_path = (
            OUT_DIR
            / 'best_unet_lnnd.pt'
        )

        for epoch in range(
            1,
            NUM_EPOCHS + 1,
        ):
            train_loss = train_one_epoch(
                model,
                train_loader,
                optimizer,
                device,
            )
            metrics = evaluate(
                model,
                val_loader,
                device,
            )

            row = {
                'epoch': epoch,
                'train_loss': train_loss,
                **metrics,
            }
            history.append(row)

            print(
                f"epoch={epoch:03d}/{NUM_EPOCHS:03d} "
                f"train_loss={train_loss:.6f} "
                f"val_unknown_rmse="
                f"{metrics['unknown_rmse']:.6f} "
                f"val_known_rmse="
                f"{metrics['known_rmse']:.6f}"
            )

            if (
                metrics['unknown_rmse']
                < best_unknown_rmse
            ):
                best_unknown_rmse = (
                    metrics['unknown_rmse']
                )

                torch.save(
                    {
                        'model_state_dict':
                            model.state_dict(),
                        'optimizer_state_dict':
                            optimizer.state_dict(),
                        'epoch': epoch,
                        'metrics': metrics,
                        'config': config,
                    },
                    best_model_path,
                )

        (
            OUT_DIR
            / 'history.json'
        ).write_text(
            json.dumps(
                history,
                indent=2,
            ),
            encoding='utf-8',
        )

        print(
            f'Saved best model: '
            f'{best_model_path}'
        )
        print(
            f'Saved training history: '
            f'{OUT_DIR / "history.json"}'
        )

        # 训练结束以后重新加载 20 个 epoch 中验证集
        # unknown RMSE 最低的那个 checkpoint，而不是直接使用 epoch 20。
        checkpoint = torch.load(
            best_model_path,
            map_location=device,
            weights_only=False,
        )
        model.load_state_dict(
            checkpoint['model_state_dict']
        )
        model.eval()

        print(
            'Best epoch within the 20 epochs: '
            f'{checkpoint["epoch"]}'
        )
        print(
            'Best validation unknown RMSE: '
            f'{checkpoint["metrics"]["unknown_rmse"]:.6f}'
        )

    # ---------------------------------------------------------
    # 从这里开始，无论是“加载已有模型”还是“重新训练”，
    # 都执行完全相同的验证误差和 counterfactual 诊断。
    # ---------------------------------------------------------

    # ---------------------------------------------------------
    # 1. VAL_YEARS 空间误差
    #    residual = lnNd_observed - lnNd_predicted
    # ---------------------------------------------------------
    print(
        'Collecting validation spatial error...'
    )
    val_acc = collect_spatial_residual_stats(
        model,
        val_loader,
        device,
    )
    val_stats = summarize_spatial_accumulator(
        val_acc
    )

    # ---------------------------------------------------------
    # 2. 目标年份 counterfactual difference
    #    delta = lnNd_observed - lnNd_counterfactual
    # ---------------------------------------------------------
    print(
        'Collecting counterfactual spatial difference...'
    )
    target_acc = collect_spatial_residual_stats(
        model,
        target_loader,
        device,
    )
    target_stats = summarize_spatial_accumulator(
        target_acc
    )

    # ---------------------------------------------------------
    # 3. 逐格点判断：
    #    |target delta| 是否显著大于 validation absolute error
    # ---------------------------------------------------------
    abs_test = (
        welch_compare_target_abs_vs_validation_abs(
            val_stats,
            target_stats,
        )
    )

    # signed delta 与 validation signed bias 是否显著不同。
    signed_test = welch_compare_signed_means(
        val_stats,
        target_stats,
    )

    lat, lon = get_grid_coordinates(
        train_paths[0]
    )

    diagnostic_nc = (
        OUT_DIR
        / 'spatial_counterfactual_diagnostics.nc'
    )
    save_spatial_diagnostics(
        diagnostic_nc,
        lat,
        lon,
        val_stats,
        target_stats,
        abs_test,
        signed_test,
    )

    diagnostic_pngs = plot_spatial_diagnostics(
        OUT_DIR,
        lat,
        lon,
        val_stats,
        target_stats,
    )

    global_summary_path = (
        OUT_DIR
        / 'global_counterfactual_summary.json'
    )
    save_global_summary(
        global_summary_path,
        val_acc,
        target_acc,
    )

    val_global = global_stats_from_accumulator(
        val_acc
    )
    target_global = global_stats_from_accumulator(
        target_acc
    )

    sig_fraction_den = np.sum(
        (
            val_stats['count']
            >= MIN_GRID_SAMPLES
        )
        & (
            target_stats['count']
            >= MIN_GRID_SAMPLES
        )
        & np.isfinite(
            abs_test['p_one_sided']
        )
    )
    sig_fraction_num = np.sum(
        abs_test['significant_larger']
    )

    sig_fraction = (
        sig_fraction_num
        / sig_fraction_den
        if sig_fraction_den > 0
        else np.nan
    )

    print('')
    print('========== FINAL DIAGNOSTICS ==========')
    print(
        'Validation pooled MAE '
        '(observed - predicted): '
        f'{val_global["mae"]:.6f}'
    )
    print(
        'Validation pooled RMSE '
        '(observed - predicted): '
        f'{val_global["rmse"]:.6f}'
    )
    print(
        'Target pooled mean '
        '(lnNd_observed - lnNd_counterfactual): '
        f'{target_global["mean"]:.6f}'
    )
    print(
        'Target pooled mean absolute difference: '
        f'{target_global["mae"]:.6f}'
    )

    if val_global['mae'] > 0:
        print(
            'Target absolute difference / '
            'validation MAE: '
            f'{target_global["mae"] / val_global["mae"]:.3f}'
        )

    print(
        'Fraction of tested grid cells where '
        '|target delta| is significantly larger '
        'than validation absolute error: '
        f'{sig_fraction:.3%}'
        if np.isfinite(sig_fraction)
        else 'No grid cells have enough samples '
        'for significance testing.'
    )

    print(
        f'Saved spatial diagnostics: '
        f'{diagnostic_nc}'
    )
    print('Saved spatial figures:')
    for figure_path in diagnostic_pngs:
        print(f'  {figure_path}')
    print(
        f'Saved global summary: '
        f'{global_summary_path}'
    )


if __name__ == '__main__':
    main()
