from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.ensemble import RandomForestRegressor
from sklearn.impute import SimpleImputer
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score
from sklearn.model_selection import train_test_split
from sklearn.pipeline import Pipeline


DATA_CSV = Path('/home/chenyiqi/260320_ship_emission/processed_data/ml_xy_data/2020/soxdiff_met_and_cld_202001021330.csv')
GAMM = 1.37e-5
RANDOM_STATE = 42
TEST_SIZE = 0.2


def _build_nd(df: pd.DataFrame) -> pd.Series:
	return GAMM * np.power(df['cot_mod08'], 0.5) * np.power(df['cer_mod08'] * 1e-6, -2.5) * 1e-6


def _add_pressure_diff_features(df: pd.DataFrame) -> pd.DataFrame:
	out = df.copy()
	bases = ['u', 'v', 'w', 'vo', 'r', 'q', 'z', 'pv', 'd', 't']
	for base in bases:
		c1000 = f'{base}_1000'
		c750 = f'{base}_750'
		c500 = f'{base}_500'
		if c1000 in out.columns and c750 in out.columns:
			out[f'{base}_1000_minus_750'] = out[c1000] - out[c750]
		if c750 in out.columns and c500 in out.columns:
			out[f'{base}_750_minus_500'] = out[c750] - out[c500]
	return out


def _select_feature_columns(df: pd.DataFrame) -> list[str]:
	exclude = {
		'weighted_sox_diff',
		'cf_ret_liq_mod08',
		'cot_mod08',
		'cer_mod08',
		'cf_ret_combined_mod08',
		'mwd',
		'mwp',
		'aod_mod08',
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


def _train_one_target(df: pd.DataFrame, feature_cols: list[str], target_col: str) -> dict[str, float | int | str]:
	target_mask = np.isfinite(df[target_col].to_numpy(dtype=float))
	work_df = df.loc[target_mask, feature_cols + [target_col]].copy()

	if work_df.empty:
		raise ValueError(f'No finite rows available for target: {target_col}')

	X = work_df[feature_cols]
	y = work_df[target_col]

	X_train, X_test, y_train, y_test = train_test_split(
		X,
		y,
		test_size=TEST_SIZE,
		random_state=RANDOM_STATE,
	)

	model = Pipeline(
		steps=[
			('imputer', SimpleImputer(strategy='median')),
			(
				'regressor',
				RandomForestRegressor(
					n_estimators=400,
					random_state=RANDOM_STATE,
					n_jobs=-1,
				),
			),
		]
	)
	model.fit(X_train, y_train)
	pred = model.predict(X_test)

	return {
		'target': target_col,
		'n_rows': int(len(work_df)),
		'n_features': int(len(feature_cols)),
		'r2': float(r2_score(y_test, pred)),
		'rmse': float(np.sqrt(mean_squared_error(y_test, pred))),
		'mae': float(mean_absolute_error(y_test, pred)),
	}


def main() -> None:
	df = pd.read_csv(DATA_CSV)
	df = _add_pressure_diff_features(df)
	df['nd'] = _build_nd(df)
	# print(np.mean(df['nd']))

	feature_cols = _select_feature_columns(df)
	targets = ['cf_ret_liq_mod08', 'nd', 'cwp_mod08']

	for target in targets:
		if target not in df.columns:
			raise ValueError(f'Missing target column in data: {target}')

	results = [_train_one_target(df, feature_cols, target) for target in targets]
	result_df = pd.DataFrame(results)

	print(f'Data CSV: {DATA_CSV}')
	print(f'Feature count: {len(feature_cols)}')
	print('Targets:', ', '.join(targets))
	print(result_df.to_string(index=False))


if __name__ == '__main__':
	main()
