from __future__ import annotations

import datetime as dt
import re
from pathlib import Path
import numpy as np
import torch
import xarray as xr
from util import (
    collect_spatial_residual_stats,
    configure_output,
    global_stats_from_accumulator,
    plot_spatial_diagnostics,
    save_global_summary,
    save_json,
    save_spatial_diagnostics,
    summarize_spatial_accumulator,
    welch_compare_signed_means,
    welch_compare_target_abs_vs_validation_abs,
)


DATA_ROOT = Path('/home/chenyiqi/260320_ship_emission/processed_data/ml_grid_data')
OUT_DIR = Path('/home/chenyiqi/260320_ship_emission/processed_data/unet_lnnd_no_M')

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
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
KNOWN_LOSS_WEIGHT = 0.05
NUM_WORKERS = 64

# ------------------------------------------------------------
# 模型使用方式：
# True  -> 直接加载已有 best_unet_lnnd.pt，跳过训练；
# False -> 从头重新训练，最多训练 NUM_EPOCHS=20 个 epoch。
# ------------------------------------------------------------
USE_SAVED_MODEL = False
SAVED_MODEL_PATH = OUT_DIR / 'pretrained_unet_lnnd.pt'

# mask-only augmentation for pretraining
PRETRAIN_AUGMENTATION = True
AUGMENT_NUMBER = 10

# Random-mask self-supervised pretraining.
# The mask is generated randomly and is not related to SOx.
RANDOM_MASK_FRACTION = 0.10

# U-Net input mode:
# 'feature'      : only feature_vars
# 'feature+lnnd' : feature_vars + lnnd_known_fill
INPUT_MODE = 'feature'

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
    """
    Fixed input variables for U-Net.

    Only use selected meteorological and MODIS variables.
    Latitude and longitude channels are not used.
    """
    selected_vars = ['ssza_mod08', 'ssaa_mod08', 'slza_mod08', 'slaa_mod08', 'd_1000', 'd_750', 'd_500', 'z_1000', 'z_750', 'z_500', 'pv_1000', 'pv_750', 'pv_500', 'r_1000', 'r_750', 'r_500', 'q_1000', 'q_750', 'q_500', 't_1000', 't_750', 't_500', 'u_1000', 'u_750', 'u_500', 'v_1000', 'v_750', 'v_500', 'w_1000', 'w_750', 'w_500', 'vo_1000', 'vo_750', 'vo_500', 'u10', 'v10', 'd2m', 't2m', 'msl', 'sst', 'sp', 'blh', 'cape', 'tp', 'lsp', 'lspf', 'bld', 'mwd', 'mwp', 'ptype']

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



def generate_random_masks(
    valid_mask: np.ndarray,
    number: int = 10,
    fraction: float = 0.10,
    seed: int | None = None,
) -> list[np.ndarray]:
    """
    Random mask generation for self-supervised pretraining.

    Only unknown_mask and lnnd_known_fill are changed.
    All meteorological/geophysical input variables remain unchanged.
    """

    rng = np.random.default_rng(seed)

    valid_indices = np.flatnonzero(
        valid_mask.ravel() > 0.5
    )

    masks = []

    n_mask = max(
        1,
        int(len(valid_indices) * fraction)
    )

    for _ in range(number):
        selected = rng.choice(
            valid_indices,
            size=n_mask,
            replace=False,
        )

        mask = np.zeros_like(
            valid_mask,
            dtype=np.float32,
        )

        mask.ravel()[selected] = 1.0
        masks.append(mask)

    return masks




class LnndGridDataset(Dataset):
    def __init__(
        self,
        paths: list[Path],
        feature_vars: list[str],
        feature_stats: dict[str, tuple[float, float]],
        augmentation: bool = False,
    ):
        self.paths = paths
        self.feature_vars = feature_vars
        self.feature_stats = feature_stats
        self.augmentation = augmentation

    def __len__(self) -> int:
        if self.augmentation:
            return len(self.paths) * AUGMENT_NUMBER
        return len(self.paths)

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        if self.augmentation:
            file_index = index // AUGMENT_NUMBER
            aug_index = index % AUGMENT_NUMBER
        else:
            file_index = index
            aug_index = 0

        path = self.paths[file_index]

        with xr.open_dataset(path) as ds:
            lnnd_true = ds[TARGET_COL].values.astype(np.float32)
            current_accu_sox = ds['accu_sox'].values.astype(np.float32)

            target_valid = np.isfinite(lnnd_true).astype(np.float32)

            # 新定义：当前样本 accu_sox 最大的 10%。
            if self.augmentation:
                # Scheme B:
                # random mask pretraining.
                # SOx is not used in the pretraining mask.
                unknown_mask = generate_random_masks(
                    target_valid,
                    number=AUGMENT_NUMBER,
                    fraction=RANDOM_MASK_FRACTION,
                    seed=index,
                )[aug_index]
            else:
                # Fine-tuning:
                # use the real SOx mask.
                unknown_mask = build_mask_from_accu_sox(
                    current_accu_sox=current_accu_sox,
                    target_valid=target_valid,
                )

            unknown_mask = (
                unknown_mask * target_valid
            ).astype(np.float32)

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

            # Do not use latitude/longitude channels.
            # Optional lnNd input. M_unknown is not used.
            if INPUT_MODE in ('lnnd', 'feature+lnnd'):
                channels.append(
                    lnnd_known.astype(np.float32)
                )

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

    configure_output(
        mask_definition=f"Top {100 * (1 - MASK_QUANTILE):.1f}% of valid lnNd grid cells",
        min_grid_samples=MIN_GRID_SAMPLES,
        significance_level=SIGNIFICANCE_LEVEL,
        validation_years=VAL_YEARS,
        counterfactual_years=COUNTERFACTUAL_YEARS,
        output_tag='pretrain',
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
        augmentation=PRETRAIN_AUGMENTATION,
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

    in_channels = len(feature_vars)

    if INPUT_MODE in ('lnnd', 'feature+lnnd'):
        in_channels += 1

    model = UNet(
        in_channels=in_channels
    ).to(device)

    config = {
        'train_years': TRAIN_YEARS,
        'val_years': VAL_YEARS,
        'counterfactual_years': COUNTERFACTUAL_YEARS,
        'mask_quantile': MASK_QUANTILE,
        'mask_definition': (
            'random masking over valid ocean lnNd pixels '
            f'({RANDOM_MASK_FRACTION * 100:.1f}% fraction)'
        ),
        'feature_vars': feature_vars,
        'input_mode': INPUT_MODE,
        'extra_channels': (
            ['feature_vars']
            if INPUT_MODE == 'feature'
            else (
                ['lnnd_known_fill']
                if INPUT_MODE == 'lnnd'
                else ['feature_vars', 'lnnd_known_fill']
            )
        ),
        'feature_stats': feature_stats,
        'target': TARGET_COL,
        'input_channels': in_channels,
        'pretrain_augmentation': PRETRAIN_AUGMENTATION,
        'augment_number': AUGMENT_NUMBER,
        'random_mask_fraction': RANDOM_MASK_FRACTION,
        'pretraining_mask': 'random_valid_ocean_mask',
        'uses_accu_sox_for_mask': False,
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
    save_json(
        OUT_DIR / 'config_pretrain.json',
        config,
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
        f'Input mode: {INPUT_MODE}'
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
            / 'pretrained_unet_lnnd.pt'
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

        save_json(
            OUT_DIR / 'history_pretrain.json',
            history,
        )

        print(
            f'Saved training history: '
            f'{OUT_DIR / "history_pretrain.json"}'
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
