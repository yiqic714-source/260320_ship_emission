from __future__ import annotations

from pathlib import Path
import numpy as np
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, Dataset
import xarray as xr
from util import (
    build_mask_from_accu_sox,
    build_raw_stack,
    cache_path_for,
    collect_grid_files,
    collect_spatial_residual_stats,
    configure_output,
    masked_loss,
    plot_spatial_diagnostics,
    prepare_preprocessed_cache,
    save_json,
    standardize,
    summarize_spatial_accumulator,
    validate_loss_type,
)


DATA_ROOT = Path('/home/chenyiqi/260320_ship_emission/processed_data/ml_grid_data')
OUT_DIR = Path('/home/chenyiqi/260320_ship_emission/project/saved_model/tuned')

# 年份设置与 pretrain_random_mask.py 保持一致（同一种定义方式、同一套年份），
# 这样预训练和微调的验证年/对比年相同，结果可以直接比较。
# VAL_YEARS 同时从训练年份里排除，避免训练/验证重叠。
VAL_YEARS = (2019,)

TRAIN_YEARS = tuple(
    year
    for year in range(2005, 2020)
    if year not in VAL_YEARS
)

# 用这些年份计算 lnNd_observed - lnNd_counterfactual。
COUNTERFACTUAL_YEARS = (2022,)

# 每个样本中，按 accu_sox 分位点取连续一段作为 unknown 区域。
# 实现（util.build_mask_from_accu_sox）与 pretrain_random_mask.py 共用：
#   pretrain: start_quantile=0.90, fraction=0.10 -> 最高 10%
#   这里    : start_quantile=0.60, fraction=0.40 -> 最高 40%
ACCU_SOX_MASK_START_QUANTILE = 0.60
ACCU_SOX_MASK_FRACTION = 0.40

TARGET_COL = 'lnnd'
RANDOM_STATE = 42
BATCH_SIZE = 2
NUM_EPOCHS = 3
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 16

# ------------------------------------------------------------
# 微调损失函数，可选（与 pretrain_random_mask.py 保持一致）：
# 'mse' -> 均方误差 (pred - target) ** 2
# 'mae' -> 平均绝对误差 |pred - target|
# 只影响微调时反向传播使用的损失；evaluate() 会同时报告 RMSE 和 MAE，
# 且 checkpoint 始终按 unknown_rmse 选择，方便不同损失之间比较。
# 注意：这里和预训练脚本的 LOSS_TYPE 是各自独立的开关。
# 具体实现（masked_loss / validate_loss_type）在 util.py 里，
# 与 pretrain_random_mask.py 共用同一份代码。
# ------------------------------------------------------------
LOSS_TYPE = 'mse'

# 与 pretrain_random_mask.py 保持一致：所有保存的文件名都带 OUTPUT_TAG，
# 这样不同参数的实验不会互相覆盖。
OUTPUT_TAG = 'sample_no19'

# ------------------------------------------------------------
# Model usage:
# True  -> load an existing fine-tuned model and skip training.
# False -> load the pretrained model and run fine-tuning.
USE_SAVED_MODEL = False
SAVED_MODEL_PATH = (
    OUT_DIR / f'tuned_unet_lnnd_{OUTPUT_TAG}.pt'
)

# 输入通道必须和预训练时一致：feature_vars + lnnd_known_fill（33 通道），
# 否则 U-Net 的 in_channels 和预训练权重对不上。
INPUT_MODE = 'feature+lnnd'

# ------------------------------------------------------------
# 预处理缓存：与 pretrain_random_mask.py 用**同一个目录**，
# 谁先跑谁建、另一个直接复用，不用再处理数据。
# 缓存里存的是原始值（feature + lnnd + accu_sox），标准化在取数时现算。
# ------------------------------------------------------------
USE_PREPROCESSED_CACHE = True
CACHE_DIR = Path(
    '/home/chenyiqi/260320_ship_emission/project/saved_model/cache_33var'
)
REBUILD_CACHE = False

# 预训练 checkpoint：目录 + tag 必须与 pretrain_random_mask.py 的
# OUT_DIR / OUTPUT_TAG 一致，否则会加载到旧模型或找不到文件。
PRETRAIN_OUT_DIR = Path(
    '/home/chenyiqi/260320_ship_emission/project/saved_model/pretrained'
)
PRETRAIN_OUTPUT_TAG = 'sample_100pct_no19_fill0'
PRETRAIN_MODEL_PATH = (
    PRETRAIN_OUT_DIR
    / f'pretrained_unet_lnnd_{PRETRAIN_OUTPUT_TAG}.pt'
)

FINETUNE_LEARNING_RATE = 1e-5
FINETUNE_EPOCHS = 4

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


def get_grid_coordinates(sample_path: Path) -> tuple[np.ndarray, np.ndarray]:
    with xr.open_dataset(sample_path) as ds:
        lat = ds['lat'].values.astype(np.float64)
        lon = ds['lon'].values.astype(np.float64)
    return lat, lon


class LnndGridDataset(Dataset):
    """
    与 pretrain_random_mask.LnndGridDataset 保持同一套数据处理：
    读缓存（或直接读 netCDF）→ 按 feature_stats 标准化 → 拼 lnnd 输入通道
    → 生成 SOx 掩膜。唯一区别是这里不做随机挖洞（没有 AUGMENT_NUMBER）。
    """

    def __init__(
        self,
        paths: list[Path],
        feature_vars: list[str],
        feature_stats: dict[str, tuple[float, float]],
    ):
        self.paths = paths
        self.feature_vars = feature_vars
        self.feature_stats = feature_stats

        # 逐通道 mean/std 预排成 (n_feature, 1, 1)，用于整块标准化。
        self.feature_means = np.asarray(
            [
                feature_stats[name][0]
                for name in feature_vars
            ],
            dtype=np.float32,
        )[:, None, None]
        self.feature_stds = np.asarray(
            [
                feature_stats[name][1]
                for name in feature_vars
            ],
            dtype=np.float32,
        )[:, None, None]

    def __len__(self) -> int:
        return len(self.paths)

    def load_raw_stack(self, path: Path) -> np.ndarray:
        """
        取该文件的原始数据数组（优先读缓存，与 pretrain 共用同一份缓存）：
            [0:n_feature] = 原始 feature 通道
            [n_feature]   = 原始 lnNd
            [n_feature+1] = 原始 accu_sox
        """
        if USE_PREPROCESSED_CACHE:
            cached = cache_path_for(CACHE_DIR, path)
            if cached.exists():
                return np.load(cached)

        return build_raw_stack(
            path,
            self.feature_vars,
        )

    def __getitem__(self, index: int) -> dict[str, torch.Tensor]:
        path = self.paths[index]
        stack = self.load_raw_stack(path)

        n_feature = len(self.feature_vars)
        lnnd_true = stack[n_feature]
        current_accu_sox = stack[n_feature + 1]

        # 与 pretrain 完全一致的逐通道标准化（整块算，nan/inf 统一填 0）。
        features = np.nan_to_num(
            (stack[:n_feature] - self.feature_means)
            / self.feature_stds,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

        target_valid = np.isfinite(lnnd_true).astype(np.float32)

        # 微调用真实 SOx 掩膜（与 pretrain 非增强模式同一口径、同一实现）。
        unknown_mask = build_mask_from_accu_sox(
            current_accu_sox=current_accu_sox,
            target_valid=target_valid,
            start_quantile=ACCU_SOX_MASK_START_QUANTILE,
            fraction=ACCU_SOX_MASK_FRACTION,
        )
        unknown_mask = (unknown_mask * target_valid).astype(np.float32)
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

        # 输入通道与 pretrain 一致：feature_vars [+ 标准化后的 lnnd_known_fill]
        if INPUT_MODE in ('lnnd', 'feature+lnnd'):
            lnnd_mean, lnnd_std = self.feature_stats[TARGET_COL]
            lnnd_channel = standardize(
                lnnd_known,
                lnnd_mean,
                lnnd_std,
            )[None, ...]
            x = np.concatenate(
                [features, lnnd_channel],
                axis=0,
            ).astype(np.float32)
        else:
            x = features

        y = np.nan_to_num(
            lnnd_true,
            nan=known_mean,
            posinf=known_mean,
            neginf=known_mean,
        )[None, ...].astype(np.float32)

        return {
            'x': torch.from_numpy(x),
            'y': torch.from_numpy(y),
            'unknown_mask': torch.from_numpy(unknown_mask[None, ...]),
            'known_mask': torch.from_numpy(known_mask[None, ...]),
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

        pred = model(x)

        # 只用 unknown 区域的损失（KNOWN_LOSS_WEIGHT 已移除，恒为 0）。
        loss = masked_loss(
            pred,
            y,
            unknown_mask,
            LOSS_TYPE,
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

    total_unknown_sq_error = 0.0
    total_known_sq_error = 0.0
    total_unknown_abs_error = 0.0
    total_known_abs_error = 0.0
    total_unknown_count = 0.0
    total_known_count = 0.0

    for batch in loader:
        x = batch['x'].to(device)
        y = batch['y'].to(device)
        unknown_mask = batch['unknown_mask'].to(device)
        known_mask = batch['known_mask'].to(device)

        pred = model(x)

        residual = pred - y

        total_unknown_sq_error += float(
            torch.sum(
                (residual ** 2) * unknown_mask
            ).cpu()
        )
        total_known_sq_error += float(
            torch.sum(
                (residual ** 2) * known_mask
            ).cpu()
        )
        total_unknown_abs_error += float(
            torch.sum(
                residual.abs() * unknown_mask
            ).cpu()
        )
        total_known_abs_error += float(
            torch.sum(
                residual.abs() * known_mask
            ).cpu()
        )

        total_unknown_count += float(
            unknown_mask.sum().cpu()
        )
        total_known_count += float(
            known_mask.sum().cpu()
        )

    unknown_mse = (
        total_unknown_sq_error
        / max(total_unknown_count, 1.0)
    )
    known_mse = (
        total_known_sq_error
        / max(total_known_count, 1.0)
    )
    unknown_mae = (
        total_unknown_abs_error
        / max(total_unknown_count, 1.0)
    )
    known_mae = (
        total_known_abs_error
        / max(total_known_count, 1.0)
    )

    # 无论 LOSS_TYPE 是 'mse' 还是 'mae'，都同时返回 RMSE 和 MAE，
    # 这样不同损失之间可以直接比较（checkpoint 仍按 unknown_rmse 选择）。
    return {
        'unknown_rmse': float(np.sqrt(unknown_mse)),
        'known_rmse': float(np.sqrt(known_mse)),
        'unknown_mse': float(unknown_mse),
        'known_mse': float(known_mse),
        'unknown_mae': float(unknown_mae),
        'known_mae': float(known_mae),
    }


def main() -> None:
    seed_everything(RANDOM_STATE)
    OUT_DIR.mkdir(
        parents=True,
        exist_ok=True,
    )

    # 提前校验损失函数名称，避免训练到一半才报错。
    loss_type = validate_loss_type(LOSS_TYPE)

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
        mask_definition=f"accu_sox quantile {ACCU_SOX_MASK_START_QUANTILE:.2f} to {ACCU_SOX_MASK_START_QUANTILE + ACCU_SOX_MASK_FRACTION:.2f}",
        validation_years=VAL_YEARS,
        counterfactual_years=COUNTERFACTUAL_YEARS,
        output_tag=OUTPUT_TAG,
    )

    # ---------------------------------------------------------
    # 特征变量与标准化参数
    #
    # 若使用已保存模型，必须优先使用 checkpoint 中保存的
    # feature_vars / feature_stats，确保输入变量、顺序和标准化方式
    # 与模型训练时完全一致。
    # ---------------------------------------------------------
    checkpoint = None

    if USE_SAVED_MODEL or PRETRAIN_MODEL_PATH.exists():
        checkpoint_path = (
            SAVED_MODEL_PATH
            if USE_SAVED_MODEL
            else PRETRAIN_MODEL_PATH
        )
        if not checkpoint_path.exists():
            raise FileNotFoundError(
                'Required model checkpoint does not exist:\n'
                f'{checkpoint_path}'
            )

        print('=' * 80)
        print('Loading saved fine-tuned model.' if USE_SAVED_MODEL else 'Loading pretrained model for fine-tuning.')
        print('Checkpoint:')
        print(checkpoint_path)
        print('=' * 80)

        checkpoint = torch.load(
            checkpoint_path,
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

        # lnnd 输入通道也要标准化，需要 lnnd 自己的统计量（与 pretrain 一致）。
        if INPUT_MODE in ('lnnd', 'feature+lnnd'):
            if TARGET_COL not in feature_stats_raw:
                raise ValueError(
                    f'The checkpoint has no stats for "{TARGET_COL}", '
                    'but INPUT_MODE requires the lnnd input channel '
                    'to be standardized. Retrain the pretrained model '
                    'with the current pretrain_random_mask.py.'
                )

            feature_stats[TARGET_COL] = tuple(
                float(v)
                for v in feature_stats_raw[TARGET_COL]
            )

    else:
        raise RuntimeError(
            'A model checkpoint is required; from-scratch training is disabled.'
        )

    # ---------------------------------------------------------
    # 预处理缓存：与 pretrain 共用同一个 CACHE_DIR，
    # 缺失的文件（例如只有微调用到的年份）在这里补齐。
    # ---------------------------------------------------------
    if USE_PREPROCESSED_CACHE:
        cache_paths = list(dict.fromkeys(
            train_paths + val_paths + target_paths
        ))
        prepare_preprocessed_cache(
            cache_paths,
            feature_vars,
            CACHE_DIR,
            REBUILD_CACHE,
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

    in_channels = len(feature_vars)
    if INPUT_MODE in ('lnnd', 'feature+lnnd'):
        in_channels += 1

    # Check compatibility with a previously saved fine-tuned model.
    if USE_SAVED_MODEL:
        saved_input_channels = checkpoint.get('config', {}).get('input_channels')
        if saved_input_channels is not None and int(saved_input_channels) != in_channels:
            raise ValueError(
                'Saved checkpoint input-channel mismatch: '
                f'checkpoint={saved_input_channels}, current={in_channels}.'
            )
    model = UNet(
        in_channels=in_channels
    ).to(device)

    config = {
        'train_years': TRAIN_YEARS,
        'val_years': VAL_YEARS,
        'counterfactual_years': COUNTERFACTUAL_YEARS,
        'mask_start_quantile': ACCU_SOX_MASK_START_QUANTILE,
        'mask_fraction': ACCU_SOX_MASK_FRACTION,
        'mask_definition': (
            f'accu_sox quantile {ACCU_SOX_MASK_START_QUANTILE:.2f} to {ACCU_SOX_MASK_START_QUANTILE + ACCU_SOX_MASK_FRACTION:.2f}'
        ),
        'input_mode': INPUT_MODE,
        'extra_channels': (
            ['lnnd_known_fill']
            if INPUT_MODE in ('lnnd', 'feature+lnnd')
            else []
        ),
        'num_epochs_if_retraining': NUM_EPOCHS,
        'loss_type': loss_type,
    }
    save_json(
        OUT_DIR / f'config_{OUTPUT_TAG}.json',
        config,
    )

    print(f'Device: {device}')
    print(f'Loss type: {loss_type}')
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
        'Mask: accu_sox quantile '
        f'{ACCU_SOX_MASK_START_QUANTILE:.2f} to {ACCU_SOX_MASK_START_QUANTILE + ACCU_SOX_MASK_FRACTION:.2f}'
    )
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
    else:
        model.load_state_dict(
            checkpoint['model_state_dict']
        )
        print('Loaded pretrained model:')
        print(PRETRAIN_MODEL_PATH)

        optimizer = torch.optim.AdamW(
            model.parameters(),
            lr=FINETUNE_LEARNING_RATE,
            weight_decay=WEIGHT_DECAY,
        )

        best_unknown_rmse = float('inf')
        history = []

        # 新训练得到的最佳模型保存在 SAVED_MODEL_PATH（文件名带 OUTPUT_TAG）。
        best_model_path = (SAVED_MODEL_PATH)

        for epoch in range(
            1,
            FINETUNE_EPOCHS + 1,
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
                f"epoch={epoch:03d}/{FINETUNE_EPOCHS:03d} "
                f"train_loss={train_loss:.6f} "
                f"val_unknown_rmse="
                f"{metrics['unknown_rmse']:.6f} "
                f"val_known_rmse="
                f"{metrics['known_rmse']:.6f} "
                f"val_unknown_mae="
                f"{metrics['unknown_mae']:.6f}"
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
            OUT_DIR / f'history_{OUTPUT_TAG}.json',
            history,
        )

        print(
            f'Saved training history: '
            f'{OUT_DIR / f"history_{OUTPUT_TAG}.json"}'
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
            'Best epoch: '
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

    lat, lon = get_grid_coordinates(
        train_paths[-1]
    )

    diagnostic_pngs = plot_spatial_diagnostics(
        OUT_DIR,
        lat,
        lon,
        val_stats,
        target_stats,
    )
    print('Saved spatial figures:')
    for figure_path in diagnostic_pngs:
        print(f'  {figure_path}')


if __name__ == '__main__':
    main()
