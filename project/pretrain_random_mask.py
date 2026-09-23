from __future__ import annotations

import os
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
    select_feature_vars,
    standardize,
    summarize_spatial_accumulator,
    validate_loss_type,
)

if not os.environ.get('DISPLAY'):
    # 无显示环境下用不弹窗的后端；这里只保存图片，不做 display。
    import matplotlib
    matplotlib.use('Agg')

import matplotlib.pyplot as plt


DATA_ROOT = Path('/home/chenyiqi/260320_ship_emission/processed_data/ml_grid_data')
OUT_DIR = Path('/home/chenyiqi/260320_ship_emission/project/saved_model/pretrained')

# 训练年份 = 2005–2019（含两端）里，排除 EXCLUDE_TRAIN_YEARS 列出的年份。
# 常见用法：
#   留一年做验证（留一法）-> 把该年份同时写进这里和 VAL_YEARS，例如 (2019,)
#   去掉数据不全的年份    -> 例如只想剔除 2014，就写 (2014,)
#   全部 2005–2019 都参与训练 -> 写 ()
VAL_YEARS = (2019,)

TRAIN_YEARS = tuple(
    year
    for year in range(2005, 2020)
    if year not in VAL_YEARS
)

# ------------------------------------------------------------
# 调参用：只随机采样一部分训练样本参与训练。
# 0.10 -> 随机抽取 10% 的训练日样本（便于快速试参数）；
# 1.0  -> 使用全部训练样本（正式训练）。
# 采样使用 RANDOM_STATE 作为随机种子，保证同一 fraction 下结果可复现。
# 注意：该子集同时用于计算 feature_stats 和训练，因此调参时的标准化
# 参数也只基于子集，正式训练时请将 TRAIN_SUBSET_FRACTION 设为 1.0。
# ------------------------------------------------------------
TRAIN_SUBSET_FRACTION = 1.
OUTPUT_TAG = 'sample_100pct_no19_fill0'

# 用这些年份计算 lnNd_observed - lnNd_counterfactual。
# 当前设置为 2020；如果以后要分析 2020–2022，可改成 (2020, 2021, 2022)。
COUNTERFACTUAL_YEARS = (2022,)

# 每个样本中，按 accu_sox 分位点取连续一段作为 unknown 区域
# （调用 util.build_mask_from_accu_sox，与 unet_finetune_pretrained.py
#  共用同一份实现，避免两套掩膜定义）：
#   start_quantile=0.90, fraction=0.10 -> 最高 10%
ACCU_SOX_MASK_START_QUANTILE = 0.90
ACCU_SOX_MASK_FRACTION = 0.10

TARGET_COL = 'lnnd'
RANDOM_STATE = 42
BATCH_SIZE = 2
NUM_EPOCHS = 7
LEARNING_RATE = 1e-4
WEIGHT_DECAY = 1e-5
NUM_WORKERS = 64

# ------------------------------------------------------------
# 训练损失函数，可选：
# 'mse' -> 均方误差 (pred - target) ** 2
# 'mae' -> 平均绝对误差 |pred - target|
# 只影响训练时反向传播使用的损失；evaluate() 会同时报告 RMSE 和 MAE，
# 且 checkpoint 始终按 unknown_rmse 选择，方便不同损失之间比较。
# 具体实现（masked_loss / validate_loss_type）在 util.py 里，
# 与 unet_finetune_pretrained.py 共用同一份代码。
# ------------------------------------------------------------
LOSS_TYPE = 'mae'
# ------------------------------------------------------------
# 模型使用方式：
# True  -> 直接加载已有 best_unet_lnnd.pt，跳过训练；
# False -> 从头重新训练，最多训练 NUM_EPOCHS=20 个 epoch。
# ------------------------------------------------------------
USE_SAVED_MODEL = False
SAVED_MODEL_PATH = OUT_DIR / f'pretrained_unet_lnnd_{OUTPUT_TAG}.pt'

# mask-only augmentation for pretraining
PRETRAIN_AUGMENTATION = True
AUGMENT_NUMBER = 10

# Random-mask self-supervised pretraining.
# The mask is generated randomly and is not related to SOx.
# 挖洞比例直接取 ACCU_SOX_MASK_FRACTION：预训练和微调的掩膜比例保持一致，
# 只有一处定义，改一个数两边都跟着变。
RANDOM_MASK_FRACTION = ACCU_SOX_MASK_FRACTION

# 随机挖洞的形态：
# 每个 mask 有 BRUSH_MASK_PROBABILITY 的概率用「笔刷」（连续笔触）生成，
# 其余情况仍是逐格点完全随机。0.0 = 全部完全随机，1.0 = 全部笔刷。
BRUSH_MASK_PROBABILITY = 0.5

# 笔刷细节：转向概率越小、笔触越直越长；8 邻域步长。
BRUSH_TURN_PROBABILITY = 0.15
BRUSH_MIN_LENGTH = 5
BRUSH_MAX_LENGTH = 60

# 笔刷粗细：笔触半径（单位 = 格点）。
#   0 = 单格宽（细笔，原来的形态）
#   1 = 3x3 圆盘
#   2 = 半径 2 的圆盘（13 格，默认）
# 每走一步就以当前位置为中心刷一个这样的圆盘，所以越大笔触越粗。
BRUSH_RADIUS = 2

BRUSH_DIRECTION_STEPS = (
    (0, 1),
    (0, -1),
    (1, 0),
    (-1, 0),
    (1, 1),
    (1, -1),
    (-1, 1),
    (-1, -1),
)

# 是否保存几张「笔刷」随机 mask 的示意图（只写 PNG，不 display）。
SAVE_MASK_EXAMPLES = True
MASK_EXAMPLE_COUNT = 2

# U-Net input mode:
# 'feature'      : only feature_vars
# 'feature+lnnd' : feature_vars + lnnd_known_fill
INPUT_MODE = 'feature+lnnd'

# ------------------------------------------------------------
# 预处理结果缓存：
# 把每个日文件预先存成 (n_feature + 2, 180, 360) 的 float32 数组（.npy）：
#     [0:n_feature] = 原始 feature 通道（顺序 = feature_vars）
#     [n_feature]   = 原始 lnNd（目标列）
#     [n_feature+1] = 原始 accu_sox（非增强模式做 SOx 掩膜要用）
# 重新训练时直接读缓存，不必再开 netCDF、再逐个变量取数（数据准备里最贵的一步）；
# 标准化在 __getitem__ 里按当前 feature_stats 现算（纯 numpy，很快）。
# 存原始值而不是标准化值：缓存只依赖 feature_vars，所以改
# TRAIN_SUBSET_FRACTION / 年份等导致统计量变化时**不用**重建缓存；
# 只有 feature_vars 变了才需要重建（脚本会报错提示，
# 设 REBUILD_CACHE=True 或删掉 CACHE_DIR 即可）。
# 体积约 8.8 MB/文件（6558 个文件 ≈ 58 GB）。
# ------------------------------------------------------------
USE_PREPROCESSED_CACHE = True
CACHE_DIR = Path(
    '/home/chenyiqi/260320_ship_emission/project/saved_model/cache_33var'
)
REBUILD_CACHE = False

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


def subsample_paths(
    paths: list[Path],
    fraction: float,
    seed: int = RANDOM_STATE,
) -> list[Path]:
    """
    从训练文件列表中随机抽取 fraction 比例的子集，用于快速调参。

    fraction >= 1.0 时返回全部文件（保持原有排序）；
    fraction <= 0.0 视为非法参数并抛出异常。
    始终至少保留 1 个文件，返回值按原顺序排列。
    """
    fraction = float(fraction)

    if fraction >= 1.0:
        return list(paths)

    if fraction <= 0.0:
        raise ValueError(
            'TRAIN_SUBSET_FRACTION must be in (0, 1], '
            f'got {fraction}'
        )

    if not paths:
        raise ValueError('Cannot subsample an empty file list.')

    n_select = max(1, int(round(fraction * len(paths))))
    n_select = min(n_select, len(paths))

    rng = np.random.default_rng(seed)
    selected = rng.choice(
        len(paths),
        size=n_select,
        replace=False,
    )

    return [paths[i] for i in np.sort(selected)]


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


def brush_walk_mask(
    rng: np.random.Generator,
    candidate_2d: np.ndarray,
    candidate_flat: np.ndarray,
    n_mask: int,
) -> np.ndarray:
    """
    用「笔刷」方式生成 mask：从随机起点出发做带方向的连续游走（一笔一笔画），
    反复画多笔，直到在候选格点内选够 n_mask 个格点。

    每一步会以当前位置为中心刷一个半径 BRUSH_RADIUS 的圆盘，
    所以笔触是有粗细的、连成一片；BRUSH_RADIUS=0 就是单格宽的细笔。

    相比逐格点独立随机，这样挖出来的洞是成片连续的，
    更接近真实缺测/云覆盖的形态（笔触之间有转向，不是死板的方块）。

    每个外层循环至少会加入起点这 1 个格点，所以一定收敛；
    圆盘内一旦刷够 n_mask 个格点就立即停止，保证格点数精确。
    """
    radius = int(BRUSH_RADIUS)
    disc_offsets = sorted(
        (
            (offset_y, offset_x)
            for offset_y in range(-radius, radius + 1)
            for offset_x in range(-radius, radius + 1)
            if offset_y * offset_y + offset_x * offset_x <= radius * radius
        ),
        key=lambda offset: offset[0] ** 2 + offset[1] ** 2,
    )

    selected = np.zeros(candidate_2d.shape, dtype=bool)
    n_y, n_x = candidate_2d.shape
    n_candidate = candidate_flat.size
    n_selected = 0
    selected_flat = selected.ravel()

    while n_selected < n_mask and n_candidate > 0:
        start_flat = int(candidate_flat[int(rng.integers(n_candidate))])
        if selected_flat[start_flat]:
            continue

        y, x = divmod(start_flat, n_x)
        dy, dx = BRUSH_DIRECTION_STEPS[
            int(rng.integers(len(BRUSH_DIRECTION_STEPS)))
        ]
        stroke_length = int(
            rng.integers(BRUSH_MIN_LENGTH, BRUSH_MAX_LENGTH + 1)
        )

        for _ in range(stroke_length):
            if n_selected >= n_mask:
                break

            # 在当前位置刷一个圆盘（笔刷粗细）。
            # offsets 已按距中心由近到远排序，(0, 0) 在最前面。
            for offset_y, offset_x in disc_offsets:
                if n_selected >= n_mask:
                    break

                paint_y = y + offset_y
                paint_x = x + offset_x
                if 0 <= paint_y < n_y and 0 <= paint_x < n_x:
                    if candidate_2d[paint_y, paint_x] and not selected[paint_y, paint_x]:
                        selected[paint_y, paint_x] = True
                        n_selected += 1

            # 小概率转向：笔触大致保持一个方向，但会缓慢拐弯。
            if rng.random() < BRUSH_TURN_PROBABILITY:
                dy, dx = BRUSH_DIRECTION_STEPS[
                    int(rng.integers(len(BRUSH_DIRECTION_STEPS)))
                ]

            new_y = y + dy
            new_x = x + dx
            if 0 <= new_y < n_y and 0 <= new_x < n_x:
                y, x = new_y, new_x
            else:
                # 走到边界就换个方向弹回来。
                dy, dx = BRUSH_DIRECTION_STEPS[
                    int(rng.integers(len(BRUSH_DIRECTION_STEPS)))
                ]
                y = int(np.clip(y + dy, 0, n_y - 1))
                x = int(np.clip(x + dx, 0, n_x - 1))

    return selected


def generate_random_masks(
    valid_mask: np.ndarray,
    number: int = 10,
    fraction: float = 0.10,
    seed: int | None = None,
    brush_probability: float | None = None,
) -> list[np.ndarray]:
    """
    Random mask generation for self-supervised pretraining.

    两种形态混合：以 brush_probability（默认 BRUSH_MASK_PROBABILITY）的概率
    用笔刷（连续笔触），否则逐格点完全随机。两种方式都只在 valid_mask 内选格点，
    且每个 mask 选到的格点数都等于 int(n_valid * fraction)。

    Only unknown_mask and lnnd_known_fill are changed.
    All meteorological/geophysical input variables remain unchanged.
    """

    rng = np.random.default_rng(seed)

    probability = (
        BRUSH_MASK_PROBABILITY
        if brush_probability is None
        else float(brush_probability)
    )

    valid_indices = np.flatnonzero(
        valid_mask.ravel() > 0.5
    )
    candidate_2d = valid_mask > 0.5

    masks = []

    n_mask = max(
        1,
        int(len(valid_indices) * fraction)
    )

    for _ in range(number):
        if rng.random() < probability:
            mask = brush_walk_mask(
                rng,
                candidate_2d,
                valid_indices,
                n_mask,
            ).astype(np.float32)
        else:
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


def save_brush_mask_examples(
    output_dir: Path,
    sample_path: Path,
    feature_vars: list[str],
    number: int = MASK_EXAMPLE_COUNT,
    seed: int = RANDOM_STATE,
) -> list[Path]:
    """
    保存几张「笔刷」随机 mask 的示意图，用来直观检查挖洞的形态。

    mask 用与训练时完全相同的 generate_random_masks() 生成，
    这里强制 brush_probability=1.0，保证展示的就是笔刷形态。

    只写 PNG 文件，不做 display（后端已在文件开头设为 Agg）。
    文件名带 OUTPUT_TAG，避免不同实验互相覆盖。
    """
    output_dir.mkdir(
        parents=True,
        exist_ok=True,
    )

    stack = build_raw_stack(
        sample_path,
        feature_vars,
    )
    lnnd_true = stack[len(feature_vars)]
    target_valid = np.isfinite(lnnd_true).astype(np.float32)

    masks = generate_random_masks(
        target_valid,
        number=number,
        fraction=RANDOM_MASK_FRACTION,
        seed=seed,
        brush_probability=1.0,
    )

    lat, lon = get_grid_coordinates(sample_path)

    saved_paths = []
    for index, mask in enumerate(masks, start=1):
        # 1 = 有效的 lnNd 格点（候选区域），2 = 被笔刷挖掉的格点。
        field = np.where(
            mask > 0.5,
            2.0,
            np.where(target_valid > 0.5, 1.0, np.nan),
        )

        fig, ax = plt.subplots(
            figsize=(10, 5),
            constrained_layout=True,
        )

        mesh = ax.pcolormesh(
            lon,
            lat,
            field,
            cmap='Blues',
            vmin=0.0,
            vmax=2.0,
            shading='auto',
        )
        colorbar = fig.colorbar(
            mesh,
            ax=ax,
            shrink=0.85,
            pad=0.03,
            ticks=[1.0, 2.0],
        )
        colorbar.ax.set_yticklabels([
            'valid lnNd cells',
            'masked (brush) cells',
        ])

        ax.set_xlabel('Longitude')
        ax.set_ylabel('Latitude')
        ax.set_title(
            f'Brush random mask {index}/{len(masks)} '
            f'({RANDOM_MASK_FRACTION * 100:.0f}% of valid cells, '
            f'{int(mask.sum())} cells, brush radius {BRUSH_RADIUS})\n'
            f'{sample_path.name}'
        )

        figure_path = (
            output_dir
            / f'brush_mask_example_{index}_{OUTPUT_TAG}.png'
        )
        fig.savefig(
            figure_path,
            dpi=250,
            bbox_inches='tight',
        )
        plt.close(fig)

        saved_paths.append(figure_path)

    return saved_paths




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
        if self.augmentation:
            return len(self.paths) * AUGMENT_NUMBER
        return len(self.paths)

    def load_raw_stack(self, path: Path) -> np.ndarray:
        """
        取该文件的原始数据数组（优先读缓存，缓存缺失时现场读 netCDF）：
            [0:n_feature] = 原始 feature 通道
            [n_feature]   = 原始 lnNd
            [n_feature+1] = 原始 accu_sox

        标准化在 __getitem__ 里按当前 feature_stats 做，所以缓存不受
        统计量变化影响（改 TRAIN_SUBSET_FRACTION / 年份等无需重建缓存）。
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
        if self.augmentation:
            file_index = index // AUGMENT_NUMBER
            aug_index = index % AUGMENT_NUMBER
        else:
            file_index = index
            aug_index = 0

        path = self.paths[file_index]
        stack = self.load_raw_stack(path)

        n_feature = len(self.feature_vars)
        lnnd_true = stack[n_feature]
        current_accu_sox = stack[n_feature + 1]

        # 按当前 feature_stats 做逐通道标准化。
        # 等价于对每个通道调用 standardize()，但这里一次性整块算完，
        # 并把 nan/inf 统一填 0（标准化后的均值）。
        features = np.nan_to_num(
            (stack[:n_feature] - self.feature_means)
            / self.feature_stds,
            nan=0.0,
            posinf=0.0,
            neginf=0.0,
        ).astype(np.float32)

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
                start_quantile=ACCU_SOX_MASK_START_QUANTILE,
                fraction=ACCU_SOX_MASK_FRACTION,
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

        # Do not use latitude/longitude channels.
        # Optional lnNd input. M_unknown is not used.
        if INPUT_MODE in ('lnnd', 'feature+lnnd'):
            # 输入通道里的 lnNd 也做标准化（与 feature 通道一致）：
            #   known 区域  = 标准化后的真实 lnNd
            #   unknown 区域 = 标准化后的「当前样本 known 区域均值」
            #   非目标有效格点 = 0（即标准化后的均值，由 standardize 处理 NaN）
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

    train_paths_all = collect_grid_files(
        DATA_ROOT,
        TRAIN_YEARS,
    )
    train_paths = subsample_paths(
        train_paths_all,
        TRAIN_SUBSET_FRACTION,
    )

    if len(train_paths) < len(train_paths_all):
        print(
            'Tuning mode: randomly sampled '
            f'{len(train_paths)}/{len(train_paths_all)} '
            f'training files '
            f'(TRAIN_SUBSET_FRACTION={TRAIN_SUBSET_FRACTION}).'
        )
    else:
        print(
            'Full training set: '
            f'{len(train_paths)} files '
            f'(TRAIN_SUBSET_FRACTION={TRAIN_SUBSET_FRACTION}).'
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

        # lnnd 输入通道也要标准化，需要 lnnd 自己的统计量。
        if INPUT_MODE in ('lnnd', 'feature+lnnd'):
            if TARGET_COL in feature_stats_raw:
                feature_stats[TARGET_COL] = tuple(
                    float(v)
                    for v in feature_stats_raw[TARGET_COL]
                )
            else:
                print(
                    f'WARNING: saved checkpoint has no stats for '
                    f'"{TARGET_COL}"; recomputing them from the '
                    'current training files.'
                )
                feature_stats[TARGET_COL] = compute_feature_stats(
                    train_paths,
                    [TARGET_COL],
                )[TARGET_COL]

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
        # lnnd 输入通道（lnnd_known_fill）也要标准化，所以把 lnnd 的统计量一起算。
        feature_stats = compute_feature_stats(
            train_paths,
            feature_vars + [TARGET_COL],
        )

    # ---------------------------------------------------------
    # 预处理缓存：把「标准化 feature + 原始 lnnd/accu_sox」存成 .npy，
    # 重新训练时直接读缓存，不必再处理数据（缺哪个补哪个）。
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
    # 笔刷随机 mask 示意图
    # ---------------------------------------------------------
    if SAVE_MASK_EXAMPLES:
        example_paths = save_brush_mask_examples(
            OUT_DIR,
            train_paths[0],
            feature_vars,
        )
        print('Saved brush mask examples:')
        for example_path in example_paths:
            print(f'  {example_path}')

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
        'mask_start_quantile': ACCU_SOX_MASK_START_QUANTILE,
        'mask_fraction': ACCU_SOX_MASK_FRACTION,
        'mask_definition': (
            'random masking over valid ocean lnNd pixels '
            f'({RANDOM_MASK_FRACTION * 100:.1f}% fraction, '
            f'brush probability {BRUSH_MASK_PROBABILITY:.2f})'
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
        'brush_mask_probability': BRUSH_MASK_PROBABILITY,
        'brush_radius': BRUSH_RADIUS,
        'save_mask_examples': SAVE_MASK_EXAMPLES,
        'mask_example_count': MASK_EXAMPLE_COUNT,
        'uses_accu_sox_for_mask': False,
        'use_preprocessed_cache': USE_PREPROCESSED_CACHE,
        'cache_dir': (
            str(CACHE_DIR)
            if USE_PREPROCESSED_CACHE
            else None
        ),
        'use_saved_model': USE_SAVED_MODEL,
        'saved_model_path': (
            str(SAVED_MODEL_PATH)
            if USE_SAVED_MODEL
            else None
        ),
        'num_epochs_if_retraining': NUM_EPOCHS,
        'loss_type': loss_type,
        'train_subset_fraction': TRAIN_SUBSET_FRACTION,
        'num_train_files_total': len(train_paths_all),
        'num_train_files_used': len(train_paths),
    }
    save_json(
        OUT_DIR / f'config_pretrain_{OUTPUT_TAG}.json',
        config,
    )

    print(f'Device: {device}')
    print(f'Loss type: {loss_type}')
    print(
        f'Train files: {len(train_paths)} '
        f'(sampled from {len(train_paths_all)}, '
        f'fraction={TRAIN_SUBSET_FRACTION}), '
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
        best_model_path = (SAVED_MODEL_PATH)

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
            OUT_DIR / f'history_pretrain_{OUTPUT_TAG}.json',
            history,
        )

        print(
            f'Saved training history: '
            f'{OUT_DIR / f"history_pretrain_{OUTPUT_TAG}.json"}'
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
            'Best epoch within the epochs: '
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
        train_paths[0]
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
