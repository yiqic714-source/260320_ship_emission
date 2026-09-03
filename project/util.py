from pathlib import Path
import json
import matplotlib.pyplot as plt
import numpy as np
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

MIN_GRID_SAMPLES = 10
SIGNIFICANCE_LEVEL = 0.05
MASK_DEFINITION = ""
VALIDATION_YEARS: Sequence[int] = ()
COUNTERFACTUAL_YEARS: Sequence[int] = ()
OUTPUT_TAG = 'output'

def configure_output(mask_definition: str, min_grid_samples: int, significance_level: float, validation_years: Sequence[int], counterfactual_years: Sequence[int], output_tag: str) -> None:
    global MASK_DEFINITION, MIN_GRID_SAMPLES, SIGNIFICANCE_LEVEL, VALIDATION_YEARS, COUNTERFACTUAL_YEARS, OUTPUT_TAG
    MASK_DEFINITION = mask_definition
    MIN_GRID_SAMPLES = min_grid_samples
    SIGNIFICANCE_LEVEL = significance_level
    VALIDATION_YEARS = tuple(validation_years)
    COUNTERFACTUAL_YEARS = tuple(counterfactual_years)
    OUTPUT_TAG = output_tag

def save_json(output_path: Path, data: object) -> None:
    output_path.write_text(
        json.dumps(data, indent=2),
        encoding='utf-8',
    )


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
                f'{MASK_DEFINITION}'
                'of current-sample accu_sox among valid lnNd grid cells.'
            ),
            'validation_years': ','.join(
                str(y) for y in VALIDATION_YEARS
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

    mean_value = area_weighted_mean(field, lat)
    ax.set_title(
        f"{title}\nArea-weighted mean: {mean_value:.4f}",
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

    1. VALIDATION_YEARS:
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
            for y in VALIDATION_YEARS
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
        / f'01_VALID_lnNd_obs_minus_predict_{OUTPUT_TAG}.png'
    )
    path_target = (
        output_dir
        / f'02_2020_lnNd_obs_minus_predict_{OUTPUT_TAG}.png'
    )
    path_difference = (
        output_dir
        / f'03_2020_minus_VALID_residual_{OUTPUT_TAG}.png'
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

    save_json(output_path, global_result)


