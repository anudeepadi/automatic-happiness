"""
Feature engineering for Port-to-Rail Surge Forecaster.
Optimized for GPU with RAPIDS cuDF/cuML.
"""

import warnings
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import DATA_CONFIG, FEATURE_COLUMNS

warnings.filterwarnings('ignore')

# Try GPU
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cudf = pd
    cp = np
    GPU_AVAILABLE = False

DataFrame = Union[pd.DataFrame, "cudf.DataFrame"]


def add_temporal_features(df: DataFrame) -> DataFrame:
    """Add comprehensive temporal features."""

    print("Adding temporal features...")

    # Convert to pandas for consistent datetime operations
    if GPU_AVAILABLE and hasattr(df, 'to_pandas'):
        pdf = df.to_pandas()
    else:
        pdf = df.copy()

    dates = pd.to_datetime(pdf['date'])

    # Basic temporal
    pdf['day_of_week'] = dates.dt.dayofweek
    pdf['month'] = dates.dt.month
    pdf['quarter'] = dates.dt.quarter
    pdf['day_of_year'] = dates.dt.dayofyear
    pdf['week_of_year'] = dates.dt.isocalendar().week.astype(int)
    pdf['day'] = dates.dt.day
    pdf['year'] = dates.dt.year

    # Boolean flags
    pdf['is_weekend'] = (pdf['day_of_week'] >= 5).astype(int)
    pdf['is_month_end'] = dates.dt.is_month_end.astype(int)
    pdf['is_quarter_end'] = dates.dt.is_quarter_end.astype(int)

    # Cyclical encoding (prevents discontinuity at boundaries)
    pdf['dow_sin'] = np.sin(2 * np.pi * pdf['day_of_week'] / 7)
    pdf['dow_cos'] = np.cos(2 * np.pi * pdf['day_of_week'] / 7)
    pdf['month_sin'] = np.sin(2 * np.pi * pdf['month'] / 12)
    pdf['month_cos'] = np.cos(2 * np.pi * pdf['month'] / 12)
    pdf['doy_sin'] = np.sin(2 * np.pi * pdf['day_of_year'] / 365)
    pdf['doy_cos'] = np.cos(2 * np.pi * pdf['day_of_year'] / 365)

    # Convert back to GPU if needed
    if GPU_AVAILABLE:
        return cudf.DataFrame(pdf)
    return pdf


def add_volume_features(df: DataFrame) -> DataFrame:
    """Add aggregated volume and compositional features."""

    print("Adding volume features...")

    if GPU_AVAILABLE and hasattr(df, 'to_pandas'):
        pdf = df.to_pandas()
    else:
        pdf = df.copy()

    # Total imports/exports
    import_cols = [c for c in pdf.columns if c.startswith('import_') and 'lag' not in c and 'ma' not in c]
    export_cols = [c for c in pdf.columns if c.startswith('export_') and 'lag' not in c and 'ma' not in c]

    pdf['total_import'] = pdf[import_cols].sum(axis=1) if import_cols else 0
    pdf['total_export'] = pdf[export_cols].sum(axis=1) if export_cols else 0
    pdf['total_volume'] = pdf['total_import'] + pdf['total_export']

    # Compositional features
    total_calls = pdf['portcalls'].replace(0, 1)
    pdf['volume_per_call'] = pdf['total_volume'] / total_calls

    # Import/export ratio
    total_export_safe = pdf['total_export'].replace(0, 1)
    pdf['import_export_ratio'] = pdf['total_import'] / total_export_safe

    # Trade balance
    pdf['trade_balance'] = pdf['total_import'] - pdf['total_export']

    # Cargo type percentages
    if 'portcalls_container' in pdf.columns:
        pdf['container_pct'] = pdf['portcalls_container'] / total_calls
    if 'portcalls_tanker' in pdf.columns:
        pdf['tanker_pct'] = pdf['portcalls_tanker'] / total_calls
    if 'portcalls_dry_bulk' in pdf.columns:
        pdf['bulk_pct'] = pdf['portcalls_dry_bulk'] / total_calls

    if GPU_AVAILABLE:
        return cudf.DataFrame(pdf)
    return pdf


def add_rolling_features(df: DataFrame, windows: Tuple[int, ...] = DATA_CONFIG.rolling_windows) -> DataFrame:
    """Add rolling statistics per port."""

    print(f"Adding rolling features (windows: {windows})...")

    if GPU_AVAILABLE and hasattr(df, 'to_pandas'):
        pdf = df.to_pandas()
    else:
        pdf = df.copy()

    # Ensure sorted
    pdf = pdf.sort_values(['portname', 'date'])

    for window in windows:
        # Moving averages
        pdf[f'ma{window}'] = pdf.groupby('portname')['portcalls'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        pdf[f'import_ma{window}'] = pdf.groupby('portname')['total_import'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )

        # Standard deviation
        pdf[f'std{window}'] = pdf.groupby('portname')['portcalls'].transform(
            lambda x: x.rolling(window, min_periods=2).std()
        ).fillna(0)

    # Z-scores for anomaly detection
    pdf['zscore_7d'] = (pdf['portcalls'] - pdf['ma7']) / pdf['std7'].replace(0, 1)
    pdf['zscore_30d'] = (pdf['portcalls'] - pdf['ma30']) / pdf['std30'].replace(0, 1)

    # Momentum (rate of change)
    for period in [3, 7, 14]:
        pdf[f'momentum_{period}d'] = pdf.groupby('portname')['portcalls'].diff(period)

    # Percent change
    pdf['pct_change'] = pdf.groupby('portname')['portcalls'].pct_change().fillna(0)
    pdf['pct_change_7d'] = pdf.groupby('portname')['portcalls'].pct_change(7).fillna(0)

    if GPU_AVAILABLE:
        return cudf.DataFrame(pdf)
    return pdf


def add_lag_features(df: DataFrame, lags: Tuple[int, ...] = DATA_CONFIG.lag_periods) -> DataFrame:
    """Add lagged features for time series prediction."""

    print(f"Adding lag features (lags: {lags})...")

    if GPU_AVAILABLE and hasattr(df, 'to_pandas'):
        pdf = df.to_pandas()
    else:
        pdf = df.copy()

    pdf = pdf.sort_values(['portname', 'date'])

    for lag in lags:
        # Port calls lag
        pdf[f'calls_lag{lag}'] = pdf.groupby('portname')['portcalls'].shift(lag)

        # Import lag
        if 'total_import' in pdf.columns:
            pdf[f'import_lag{lag}'] = pdf.groupby('portname')['total_import'].shift(lag)

        # Surge lag
        if 'surge_2std' in pdf.columns:
            pdf[f'surge_lag{lag}'] = pdf.groupby('portname')['surge_2std'].shift(lag)

    if GPU_AVAILABLE:
        return cudf.DataFrame(pdf)
    return pdf


def add_surge_indicators(
    df: DataFrame,
    zscore_threshold: float = DATA_CONFIG.surge_zscore_threshold,
    relative_threshold: float = DATA_CONFIG.surge_relative_threshold
) -> DataFrame:
    """Add surge detection indicators."""

    print("Adding surge indicators...")

    if GPU_AVAILABLE and hasattr(df, 'to_pandas'):
        pdf = df.to_pandas()
    else:
        pdf = df.copy()

    # Z-score based surge (statistical anomaly)
    pdf['surge_2std'] = (pdf['zscore_7d'] > zscore_threshold).astype(int)
    pdf['surge_3std'] = (pdf['zscore_7d'] > 3.0).astype(int)

    # Relative surge (percentage above moving average)
    pdf['surge_relative'] = (pdf['portcalls'] > pdf['ma7'] * relative_threshold).astype(int)

    # Volume surge
    if 'total_volume' in pdf.columns and 'import_ma7' in pdf.columns:
        pdf['volume_surge'] = (pdf['total_volume'] > pdf['import_ma7'] * 2).astype(int)

    if GPU_AVAILABLE:
        return cudf.DataFrame(pdf)
    return pdf


def add_port_characteristics(df: DataFrame) -> DataFrame:
    """Add port-level characteristics and ranking."""

    print("Adding port characteristics...")

    if GPU_AVAILABLE and hasattr(df, 'to_pandas'):
        pdf = df.to_pandas()
    else:
        pdf = df.copy()

    # Calculate port rankings based on average activity
    port_stats = pdf.groupby('portname').agg({
        'portcalls': 'mean',
        'total_volume': 'mean' if 'total_volume' in pdf.columns else 'portcalls'
    }).reset_index()

    port_stats['activity_rank'] = port_stats['portcalls'].rank(ascending=False, method='dense')
    port_stats['volume_rank'] = port_stats['total_volume'].rank(ascending=False, method='dense')

    # Port tiers based on activity
    n_ports = len(port_stats)
    port_stats['port_tier'] = pd.cut(
        port_stats['activity_rank'],
        bins=[0, n_ports * 0.1, n_ports * 0.3, n_ports * 0.6, n_ports + 1],
        labels=[1, 2, 3, 4]  # Tier 1 = highest activity
    ).astype(int)

    # Merge back
    pdf = pdf.merge(
        port_stats[['portname', 'activity_rank', 'volume_rank', 'port_tier']],
        on='portname',
        how='left'
    )

    if GPU_AVAILABLE:
        return cudf.DataFrame(pdf)
    return pdf


def add_target_variables(df: DataFrame) -> DataFrame:
    """Add prediction targets."""

    print("Adding target variables...")

    if GPU_AVAILABLE and hasattr(df, 'to_pandas'):
        pdf = df.to_pandas()
    else:
        pdf = df.copy()

    pdf = pdf.sort_values(['portname', 'date'])

    # Forecast targets (shift negative = future values)
    pdf['target_calls_1d'] = pdf.groupby('portname')['portcalls'].shift(-1)
    pdf['target_calls_3d'] = pdf.groupby('portname')['portcalls'].shift(-3)
    pdf['target_calls_7d'] = pdf.groupby('portname')['portcalls'].shift(-7)

    # Import forecast
    if 'total_import' in pdf.columns:
        pdf['target_import_1d'] = pdf.groupby('portname')['total_import'].shift(-1)

    # Surge forecast
    if 'surge_2std' in pdf.columns:
        pdf['target_surge_1d'] = pdf.groupby('portname')['surge_2std'].shift(-1)

    # Truck demand forecast (60% rail share, 2 TEU per truck)
    if 'target_import_1d' in pdf.columns:
        pdf['target_trucks_1d'] = pdf['target_import_1d'] * 0.6 / 2

    if GPU_AVAILABLE:
        return cudf.DataFrame(pdf)
    return pdf


def engineer_all_features(
    df: DataFrame,
    add_targets: bool = True
) -> DataFrame:
    """Run complete feature engineering pipeline."""

    print("=" * 60)
    print("Running feature engineering pipeline...")
    print("=" * 60)

    # Pipeline steps
    df = add_temporal_features(df)
    df = add_volume_features(df)
    df = add_rolling_features(df)
    df = add_surge_indicators(df)
    df = add_lag_features(df)
    df = add_port_characteristics(df)

    if add_targets:
        df = add_target_variables(df)

    # Replace infinities
    if GPU_AVAILABLE and hasattr(df, 'to_pandas'):
        pdf = df.to_pandas()
        pdf = pdf.replace([np.inf, -np.inf], np.nan)
        df = cudf.DataFrame(pdf)
    else:
        df = df.replace([np.inf, -np.inf], np.nan)

    print("=" * 60)
    print(f"✅ Feature engineering complete: {len(df.columns)} columns")
    print("=" * 60)

    return df


def get_feature_columns(df: DataFrame) -> List[str]:
    """Get available feature columns from the dataframe."""

    if GPU_AVAILABLE and hasattr(df, 'to_pandas'):
        cols = df.to_pandas().columns.tolist()
    else:
        cols = df.columns.tolist()

    # Filter to valid features
    features = [c for c in FEATURE_COLUMNS if c in cols]

    # Add any numeric columns not in the exclude list
    exclude = ['date', 'portid', 'portname', 'country', 'continent', 'year']
    exclude += [c for c in cols if c.startswith('target_')]

    for col in cols:
        if col not in features and col not in exclude:
            # Check if numeric
            if GPU_AVAILABLE and hasattr(df, 'to_pandas'):
                dtype = df.to_pandas()[col].dtype
            else:
                dtype = df[col].dtype
            if np.issubdtype(dtype, np.number):
                features.append(col)

    return list(set(features))


def prepare_ml_data(
    df: DataFrame,
    target_col: str = 'target_calls_1d',
    feature_cols: Optional[List[str]] = None
) -> Tuple[pd.DataFrame, pd.Series]:
    """Prepare X, y for model training."""

    print(f"Preparing ML data for target: {target_col}")

    if GPU_AVAILABLE and hasattr(df, 'to_pandas'):
        pdf = df.to_pandas()
    else:
        pdf = df.copy()

    # Drop rows with missing target
    pdf = pdf.dropna(subset=[target_col])

    # Get features
    if feature_cols is None:
        feature_cols = get_feature_columns(pdf)

    # Filter to available columns
    feature_cols = [c for c in feature_cols if c in pdf.columns]

    X = pdf[feature_cols].fillna(0)
    y = pdf[target_col]

    print(f"  Features: {len(feature_cols)}")
    print(f"  Samples: {len(X):,}")

    return X, y
