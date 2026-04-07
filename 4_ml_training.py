from __future__ import annotations

import datetime as dt
import re
from pathlib import Path

import matplotlib.pyplot as plt
import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_ROOT = Path('/home/chenyiqi/260320_ship_emission/processed_data/ml_xy_data')
GAMM = 1.37e-5
RANDOM_STATE = 42
TOP_FRAC = 0.10
VAL_FRAC_2020 = 0.10
YEARS = (2019, 2020)
JJA_MONTHS = {6, 7, 8}
SOX_COL = 'weighted_sox_diff'
PLOT_DIR = Path('/home/chenyiqi/260320_ship_emission/processed_data/ml_xy_data/training_figs')


def _build_nd(df: pd.DataFrame) -> pd.Series:
	return GAMM * np.power(df['cot_mod08'], 0.5) * np.power(df['cer_mod08'] * 1e-6, -2.5) * 1e-6


def _convert_pressure_diff_features(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	bases = ['u', 'v', 'w', 'vo', 'r', 'q', 'z', 'pv', 'd', 't']
	pressure_levels = [1000, 925, 850, 750, 650, 500]
	for base in bases:
		c1000 = f'{base}_1000'
		c750 = f'{base}_750'
		c500 = f'{base}_500'
		if c1000 in out.columns and c750 in out.columns:
			out[f'{base}_1000_minus_750'] = out[c1000] - out[c750]
		if c750 in out.columns and c500 in out.columns:
			out[f'{base}_750_minus_500'] = out[c750] - out[c500]

	# Remove original pressure-level columns after creating difference features.
	drop_cols = [f'{base}_{p}' for base in bases for p in pressure_levels if f'{base}_{p}' in out.columns]
	if drop_cols:
		out = out.drop(columns=drop_cols)
	return out


def _extract_date_from_name(path: Path) -> dt.date | None:
	# Expected tail in filename: *_YYYYMMDD1330.csv
	m = re.search(r'_(\d{8})\d{4}\.csv$', path.name)
	if not m:
		return None
	return dt.datetime.strptime(m.group(1), '%Y%m%d').date()


def _load_jja_data(data_root: Path) -> pd.DataFrame:
	paths = sorted(data_root.glob('*/soxdiff_met_and_cld_*.csv'))
	frames: list[pd.DataFrame] = []
	for path in paths:
		date_value = _extract_date_from_name(path)
		if date_value is None:
			continue
		if date_value.year not in YEARS or date_value.month not in JJA_MONTHS:
			continue
		df = pd.read_csv(path)
		df['source_year'] = date_value.year
		frames.append(df)

	if not frames:
		raise ValueError('No JJA CSV files found for 2019/2020.')

	return pd.concat(frames, ignore_index=True)


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
	exclude = {
		'weighted_sox_diff',
		'cf_ret_liq_mod08',
		'cot_mod08',
		'cer_mod08',
		'cwp_mod08',
		'nd',
		'cf_ret_combined_mod08',
		'aod_mod08',
		'source_year',
	}

	numeric_cols = df.select_dtypes(include=[np.number]).columns.tolist()
	feature_cols = []
	for col in numeric_cols:
		if col in exclude:
			continue
		# Exclude *_0 style columns (not pressure-level columns like *_1000).
		if col.endswith('_0'):
			continue
		feature_cols.append(col)
	return feature_cols


def _train_one_target(
	train_df: pd.DataFrame,
	val_df: pd.DataFrame,
	test_df: pd.DataFrame,
	feature_cols: list[str],
	target_col: str,
) -> tuple[dict[str, float | int | str], pd.DataFrame]:
	train_mask = np.isfinite(train_df[target_col].to_numpy(dtype=float))
	val_mask = np.isfinite(val_df[target_col].to_numpy(dtype=float))
	test_mask = np.isfinite(test_df[target_col].to_numpy(dtype=float))

	work_train = train_df.loc[train_mask, feature_cols + [target_col]].copy()
	work_val = val_df.loc[val_mask, feature_cols + [target_col]].copy()
	work_test = test_df.loc[test_mask, feature_cols + [target_col]].copy()

	if work_train.empty or work_val.empty or work_test.empty:
		raise ValueError(f'Insufficient finite rows for target: {target_col}')
	if 't0_lat' not in work_test.columns or 't0_lon' not in work_test.columns:
		raise ValueError('Test data must contain t0_lat and t0_lon for global plotting.')

	X_train = work_train[feature_cols]
	y_train = work_train[target_col]
	X_val = work_val[feature_cols]
	y_val = work_val[target_col]
	X_test = work_test[feature_cols]
	y_test = work_test[target_col]

	model = Pipeline(
		steps=[
			('imputer', SimpleImputer(strategy='median')),
			(
				'regressor',
				RandomForestRegressor(
					n_estimators=96,
					random_state=RANDOM_STATE,
					n_jobs=48,
				),
			),
		]
	)
	model.fit(X_train, y_train)
	pred_val = model.predict(X_val)
	pred_test = model.predict(X_test)
	test_plot_df = work_test[['t0_lat', 't0_lon', target_col]].copy()
	test_plot_df['test_pred'] = pred_test
	test_plot_df = (
		test_plot_df
		.groupby(['t0_lat', 't0_lon'], as_index=False)
		.agg(test_target_mean=(target_col, 'mean'), test_pred_mean=('test_pred', 'mean'))
	)

	metrics = {
		'target': target_col,
		'n_rows': int(len(work_train) + len(work_val) + len(work_test)),
		'n_features': int(len(feature_cols)),
		'r2': float(r2_score(y_val, pred_val)),
		'rmse': float(np.sqrt(mean_squared_error(y_val, pred_val))),
		'mae': float(mean_absolute_error(y_val, pred_val)),
		'val_target_mean': float(np.mean(y_val)),
		'val_pred_mean': float(np.mean(pred_val)),
		'test_target_mean': float(np.mean(y_test)),
		'test_pred_mean': float(np.mean(pred_test)),
	}
	return metrics, test_plot_df


def _plot_test_global_distribution(test_plot_df: pd.DataFrame, target_col: str, out_dir: Path) -> Path:
	out_dir.mkdir(parents=True, exist_ok=True)
	out_path = out_dir / f'test_global_{target_col}.png'

	fig, axes = plt.subplots(1, 2, figsize=(14, 5), dpi=220, constrained_layout=True)
	plot_pairs = [
		('test_target_mean', f'Test Mean of {target_col}'),
		('test_pred_mean', f'Test Mean Prediction of {target_col}'),
	]

	v1 = test_plot_df['test_target_mean'].to_numpy(dtype=float)
	v2 = test_plot_df['test_pred_mean'].to_numpy(dtype=float)
	vall = np.concatenate([v1[np.isfinite(v1)], v2[np.isfinite(v2)]])
	vmin = float(np.nanmin(vall)) if vall.size else 0.0
	vmax = float(np.nanmax(vall)) if vall.size else 1.0

	for ax, (col_name, title) in zip(axes, plot_pairs):
		sc = ax.scatter(
			test_plot_df['t0_lon'].to_numpy(dtype=float),
			test_plot_df['t0_lat'].to_numpy(dtype=float),
			c=test_plot_df[col_name].to_numpy(dtype=float),
			s=9,
			cmap='viridis',
			vmin=vmin,
			vmax=vmax,
			linewidths=0,
			alpha=0.9,
		)
		ax.set_title(title)
		ax.set_xlabel('Longitude')
		ax.set_ylabel('Latitude')
		ax.set_xlim(-180, 180)
		ax.set_ylim(-90, 90)
		ax.grid(True, linestyle='--', alpha=0.3)
		fig.colorbar(sc, ax=ax, fraction=0.045, pad=0.03)

	fig.savefig(out_path, bbox_inches='tight')
	plt.close(fig)
	return out_path


def main() -> None:
	df = _load_jja_data(DATA_ROOT)
	if SOX_COL not in df.columns:
		raise ValueError(f'Missing soxdiff column: {SOX_COL}')
	df = df[np.isfinite(df[SOX_COL].to_numpy(dtype=float))].copy()
	if df.empty:
		raise ValueError('No finite rows in soxdiff column.')

	threshold = float(df[SOX_COL].quantile(1.0 - TOP_FRAC))
	df_top = df[df[SOX_COL] >= threshold].copy()
	if df_top.empty:
		raise ValueError('No rows selected in top soxdiff fraction.')

	df_2019 = df_top[df_top['source_year'] == 2019].copy()
	df_2020 = df_top[df_top['source_year'] == 2020].copy()
	# df_2019 = _convert_pressure_diff_features(df_2019)
	df_2019['nd'] = _build_nd(df_2019)
	# df_2020 = _convert_pressure_diff_features(df_2020)
	df_2020['nd'] = _build_nd(df_2020)
	if df_2019.empty or df_2020.empty:
		raise ValueError('Top soxdiff rows do not contain both 2019 and 2020 data.')

	train_2020, val_2020 = train_test_split(
		df_2020,
		test_size=VAL_FRAC_2020,
		random_state=RANDOM_STATE,
	)

	train_df = train_2020.reset_index(drop=True)
	val_df = val_2020.reset_index(drop=True)
	test_df = df_2019.reset_index(drop=True)

	feature_cols = _select_feature_columns(pd.concat([train_df, val_df, test_df], axis=0, ignore_index=True))
	targets = ['cf_ret_liq_mod08', 'nd', 'cwp_mod08']

	for target in targets:
		if target not in train_df.columns:
			raise ValueError(f'Missing target column in data: {target}')

	print(f'Selected period: 2019/2020 JJA')
	print(f'Rows -> loaded finite soxdiff: {len(df)}, top{TOP_FRAC:.0%}: {len(df_top)}')
	print(f'Rows -> train(2020): {len(train_df)}, val(2020): {len(val_df)}, test(2019): {len(test_df)}')
	print(f'Feature count: {len(feature_cols)}')
	print('Features: ' + ', '.join(feature_cols))
	print('Targets:', ', '.join(targets))
	print('Start training...')

	results = []
	for target in targets:
		metrics, test_plot_df = _train_one_target(train_df, val_df, test_df, feature_cols, target)
		plot_path = _plot_test_global_distribution(test_plot_df, target, PLOT_DIR)
		print(f'Saved test global plot ({target}): {plot_path}')
		results.append(metrics)
	result_df = pd.DataFrame(results)

	print(f'Top fraction: {TOP_FRAC:.0%}, threshold={threshold:.6g}')
	print(result_df.to_string(index=False))


if __name__ == '__main__':
	main()
