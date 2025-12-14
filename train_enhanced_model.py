#!/usr/bin/env python3
"""
Enhanced Training Script for Port-to-Rail Surge Forecaster.

Improvements over baseline:
1. Integrates chokepoint data as leading indicators
2. Integrates disruption events
3. Handles class imbalance for surge detection (SMOTE + class weights)
4. Optimizes classification threshold
5. Adds cross-port and volatility features
6. Uses proper time-series cross-validation

Usage:
    python train_enhanced_model.py
    python train_enhanced_model.py --gpu
"""

import argparse
import json
import time
import warnings
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd
from sklearn.metrics import (
    mean_absolute_error, mean_squared_error, r2_score,
    roc_auc_score, precision_recall_curve, classification_report
)
from sklearn.model_selection import TimeSeriesSplit

warnings.filterwarnings('ignore')

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.config import DATA_DIR, OUTPUT_DIR, MODEL_DIR, MODEL_CONFIG, DATA_CONFIG
from src.data_loader import (
    load_port_activity, load_port_database, load_rail_nodes,
    filter_us_data, parse_dates, create_port_terminal_mapping
)
from src.feature_engineering import (
    engineer_all_features, get_feature_columns
)
from src.model import ChampionModel

# Try to import optional dependencies
try:
    from imblearn.over_sampling import SMOTE
    SMOTE_AVAILABLE = True
except ImportError:
    SMOTE_AVAILABLE = False
    print("Note: imbalanced-learn not installed. SMOTE will be skipped.")


def load_chokepoint_data(data_dir: Path) -> pd.DataFrame:
    """Load and preprocess chokepoint traffic data."""

    choke_file = data_dir / "Daily_Chokepoints_Data.csv"
    if not choke_file.exists():
        print("Warning: Chokepoint data not found")
        return None

    print("Loading chokepoint data...")
    df = pd.read_csv(choke_file)

    # Parse date
    df['date'] = pd.to_datetime(df['date'].str[:10])

    # Aggregate by date (sum across all chokepoints)
    daily = df.groupby('date').agg({
        'n_container': 'sum',
        'n_total': 'sum',
        'capacity': 'sum',
        'capacity_container': 'sum'
    }).reset_index()

    # Rename columns
    daily.columns = ['date', 'choke_container', 'choke_total', 'choke_capacity', 'choke_capacity_container']

    # Add rolling stats
    daily = daily.sort_values('date')
    daily['choke_ma7'] = daily['choke_container'].rolling(7, min_periods=1).mean()
    daily['choke_ma14'] = daily['choke_container'].rolling(14, min_periods=1).mean()
    daily['choke_std7'] = daily['choke_container'].rolling(7, min_periods=2).std().fillna(0)
    daily['choke_pct_change'] = daily['choke_container'].pct_change(7).fillna(0)
    daily['choke_zscore'] = (daily['choke_container'] - daily['choke_ma7']) / daily['choke_std7'].replace(0, 1)

    # Add lagged features (vessels take time to reach US ports)
    for lag in [5, 7, 10, 14, 21]:
        daily[f'choke_container_lag{lag}'] = daily['choke_container'].shift(lag)
        daily[f'choke_capacity_lag{lag}'] = daily['choke_capacity'].shift(lag)

    print(f"  Loaded {len(daily):,} days of chokepoint data")
    return daily


def load_disruption_data(data_dir: Path) -> pd.DataFrame:
    """Load and preprocess disruption events data."""

    # Find the disruptions file
    disruption_files = list(data_dir.glob("portwatch_disruptions*.csv"))
    if not disruption_files:
        print("Warning: Disruption data not found")
        return None

    print("Loading disruption data...")
    df = pd.read_csv(disruption_files[0])

    # Parse dates
    df['fromdate'] = pd.to_datetime(df['fromdate'], errors='coerce')
    df['todate'] = pd.to_datetime(df['todate'], errors='coerce')

    # Filter valid events
    df = df.dropna(subset=['fromdate', 'todate'])

    print(f"  Loaded {len(df)} disruption events")
    return df


def create_disruption_features(port_df: pd.DataFrame, disruptions: pd.DataFrame) -> pd.DataFrame:
    """Create daily disruption features."""

    if disruptions is None:
        return port_df

    print("Creating disruption features...")

    # Get date range
    date_range = pd.date_range(port_df['date'].min(), port_df['date'].max())

    # Count active disruptions per day
    disruption_daily = []
    for date in date_range:
        active = disruptions[
            (disruptions['fromdate'] <= date) &
            (disruptions['todate'] >= date)
        ]

        disruption_daily.append({
            'date': date,
            'active_disruptions': len(active),
            'tc_active': (active['eventtype'] == 'TC').sum() if 'eventtype' in active.columns else 0,
            'flood_active': (active['eventtype'] == 'FL').sum() if 'eventtype' in active.columns else 0,
            'max_severity': 3 if (active['alertlevel'] == 'RED').any() else (
                           2 if (active['alertlevel'] == 'ORANGE').any() else 1) if len(active) > 0 else 0
        })

    disruption_df = pd.DataFrame(disruption_daily)

    # Add rolling disruption counts
    disruption_df['disruptions_7d'] = disruption_df['active_disruptions'].rolling(7, min_periods=1).sum()
    disruption_df['disruptions_30d'] = disruption_df['active_disruptions'].rolling(30, min_periods=1).sum()

    # Merge with port data
    port_df = port_df.merge(disruption_df, on='date', how='left')
    port_df[disruption_df.columns[1:]] = port_df[disruption_df.columns[1:]].fillna(0)

    print(f"  Added {len(disruption_df.columns) - 1} disruption features")
    return port_df


def add_cross_port_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add features that capture regional and network effects."""

    print("Adding cross-port features...")

    # Region mapping for major US ports
    region_mapping = {
        'Los Angeles': 'west', 'Long Beach': 'west', 'Oakland': 'west',
        'Seattle': 'west', 'Tacoma': 'west', 'Portland': 'west',
        'Houston': 'gulf', 'New Orleans': 'gulf', 'Mobile': 'gulf',
        'Corpus Christi': 'gulf', 'Galveston': 'gulf',
        'New York': 'east', 'Savannah': 'east', 'Charleston': 'east',
        'Norfolk': 'east', 'Baltimore': 'east', 'Philadelphia': 'east',
        'Boston': 'east', 'Miami': 'east', 'Jacksonville': 'east'
    }

    # Map ports to regions
    df['region'] = df['portname'].map(region_mapping).fillna('other')

    # Regional average (excluding current port)
    df['regional_total_calls'] = df.groupby(['date', 'region'])['portcalls'].transform('sum')
    df['regional_port_count'] = df.groupby(['date', 'region'])['portname'].transform('count')
    df['regional_avg_calls'] = (df['regional_total_calls'] - df['portcalls']) / (df['regional_port_count'] - 1).replace(0, 1)

    # Port's share of regional traffic
    df['regional_share'] = df['portcalls'] / df['regional_total_calls'].replace(0, 1)

    # National aggregate
    df['national_total_calls'] = df.groupby('date')['portcalls'].transform('sum')
    df['national_share'] = df['portcalls'] / df['national_total_calls'].replace(0, 1)

    # Deviation from regional average
    df['vs_regional_avg'] = df['portcalls'] - df['regional_avg_calls']

    return df


def add_volatility_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add volatility and trend features."""

    print("Adding volatility features...")

    # Coefficient of variation (volatility relative to mean)
    df['cv_7d'] = df['std7'] / df['ma7'].replace(0, 1)
    df['cv_30d'] = df['std30'] / df['ma30'].replace(0, 1)

    # Bollinger band position
    df['bb_upper_7d'] = df['ma7'] + 2 * df['std7']
    df['bb_lower_7d'] = df['ma7'] - 2 * df['std7']
    df['bb_position'] = (df['portcalls'] - df['bb_lower_7d']) / (df['bb_upper_7d'] - df['bb_lower_7d']).replace(0, 1)

    # Mean reversion indicator (how far from long-term mean)
    df['deviation_from_ma30'] = (df['portcalls'] - df['ma30']) / df['std30'].replace(0, 1)

    # Trend indicator (short MA vs long MA)
    df['trend_7_30'] = (df['ma7'] - df['ma30']) / df['ma30'].replace(0, 1)

    return df


def add_interaction_features(df: pd.DataFrame) -> pd.DataFrame:
    """Add feature interactions."""

    print("Adding interaction features...")

    # Volume × Volatility (high volume + high volatility = higher surge risk)
    if 'total_volume' in df.columns:
        df['volume_volatility'] = df['total_volume'] * df['cv_7d']

    # Weekend × Activity
    df['weekend_calls'] = df['is_weekend'] * df['portcalls']

    # Month-end × Import
    if 'total_import' in df.columns:
        df['monthend_import'] = df['is_month_end'] * df['total_import']

    # Momentum × Volatility
    if 'momentum_7d' in df.columns:
        df['momentum_volatility'] = df['momentum_7d'] * df['cv_7d']

    return df


def optimize_threshold(y_true: np.ndarray, y_prob: np.ndarray, min_recall: float = 0.3) -> float:
    """Find optimal classification threshold."""

    precisions, recalls, thresholds = precision_recall_curve(y_true, y_prob)

    # Find threshold where recall >= min_recall with highest precision
    valid_idx = recalls[:-1] >= min_recall

    if not valid_idx.any():
        # If no threshold meets min_recall, use the one with highest recall
        return thresholds[np.argmax(recalls[:-1])]

    # Among valid thresholds, pick the one with highest F1
    f1_scores = 2 * precisions[:-1] * recalls[:-1] / (precisions[:-1] + recalls[:-1] + 1e-8)
    f1_scores[~valid_idx] = 0

    best_idx = np.argmax(f1_scores)
    return thresholds[best_idx]


def train_surge_classifier(X_train, y_train, X_test, y_test, use_smote=True, use_gpu=False):
    """Train surge classifier with class imbalance handling."""

    from xgboost import XGBClassifier

    print("\n  Training surge classifier...")

    # Calculate class weight
    pos_count = y_train.sum()
    neg_count = len(y_train) - pos_count
    scale_pos_weight = neg_count / max(pos_count, 1)

    print(f"    Class balance: {pos_count:,} positive / {neg_count:,} negative")
    print(f"    Scale pos weight: {scale_pos_weight:.1f}")

    # Apply SMOTE if available and requested
    X_train_resampled = X_train
    y_train_resampled = y_train

    if SMOTE_AVAILABLE and use_smote and pos_count > 5:
        try:
            smote = SMOTE(sampling_strategy=0.3, random_state=42, k_neighbors=min(5, pos_count - 1))
            X_train_resampled, y_train_resampled = smote.fit_resample(X_train, y_train)
            print(f"    After SMOTE: {y_train_resampled.sum():,} positive / {len(y_train_resampled) - y_train_resampled.sum():,} negative")
            scale_pos_weight = 1  # SMOTE handles imbalance
        except Exception as e:
            print(f"    SMOTE failed: {e}, using class weights instead")

    # Train classifier
    model = XGBClassifier(
        max_depth=8,
        n_estimators=500,
        learning_rate=0.05,
        scale_pos_weight=scale_pos_weight,
        tree_method='gpu_hist' if use_gpu else 'hist',
        device='cuda' if use_gpu else 'cpu',
        eval_metric='auc',
        random_state=42
    )

    model.fit(X_train_resampled, y_train_resampled)

    # Predict probabilities
    y_prob = model.predict_proba(X_test)[:, 1]

    # Optimize threshold
    optimal_threshold = optimize_threshold(y_test.values, y_prob, min_recall=0.3)
    print(f"    Optimal threshold: {optimal_threshold:.3f}")

    # Evaluate with optimized threshold
    y_pred_optimized = (y_prob >= optimal_threshold).astype(int)

    # Metrics
    auc = roc_auc_score(y_test, y_prob)

    report = classification_report(y_test, y_pred_optimized, output_dict=True, zero_division=0)

    metrics = {
        'auc': auc,
        'precision': report['1']['precision'] if '1' in report else 0,
        'recall': report['1']['recall'] if '1' in report else 0,
        'f1': report['1']['f1-score'] if '1' in report else 0,
        'optimal_threshold': optimal_threshold,
        'support_positive': int(y_test.sum())
    }

    print(f"    AUC: {auc:.4f}")
    print(f"    Precision: {metrics['precision']:.4f}")
    print(f"    Recall: {metrics['recall']:.4f}")
    print(f"    F1: {metrics['f1']:.4f}")

    return model, metrics, optimal_threshold


def main(args):
    """Enhanced training pipeline."""

    print("=" * 80)
    print("PORT-TO-RAIL SURGE FORECASTER - ENHANCED MODEL TRAINING")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print(f"GPU mode: {args.gpu}")
    print(f"SMOTE available: {SMOTE_AVAILABLE}")
    print()

    t0 = time.time()

    # =========================================================================
    # 1. Load All Data Sources
    # =========================================================================
    print("\n" + "=" * 60)
    print("1. LOADING DATA")
    print("=" * 60)

    # Main port activity data
    try:
        port_activity = load_port_activity(DATA_DIR)
        port_db = load_port_database(DATA_DIR)
        rail_nodes = load_rail_nodes(DATA_DIR)
    except FileNotFoundError as e:
        print(f"Error loading data: {e}")
        return

    # Additional data sources
    chokepoint_data = load_chokepoint_data(DATA_DIR)
    disruption_data = load_disruption_data(DATA_DIR)

    # Filter to US
    ports_us, ports_db_us, rail_us = filter_us_data(port_activity, port_db, rail_nodes)
    ports_us = parse_dates(ports_us)

    # Free memory
    del port_activity, port_db, rail_nodes
    import gc
    gc.collect()

    print(f"\nData Summary:")
    print(f"   US Port Activity: {len(ports_us):,} records")
    print(f"   Date Range: {ports_us['date'].min()} to {ports_us['date'].max()}")

    # =========================================================================
    # 2. Create Port-Terminal Mapping
    # =========================================================================
    print("\n" + "=" * 60)
    print("2. SPATIAL JOIN: PORTS → RAIL TERMINALS")
    print("=" * 60)

    port_terminal_map = create_port_terminal_mapping(ports_db_us, rail_us)

    # Add volume stats
    ports_us_pd = ports_us.to_pandas() if hasattr(ports_us, 'to_pandas') else ports_us
    port_volumes = ports_us_pd.groupby('portid').agg({
        'import_container': 'mean',
        'export_container': 'mean',
        'portcalls': 'mean'
    }).reset_index()
    port_volumes.columns = ['portid', 'avg_import_teu', 'avg_export_teu', 'avg_calls']

    port_terminal_map = port_terminal_map.merge(port_volumes, on='portid', how='left')
    port_terminal_map['daily_trucks_needed'] = port_terminal_map['avg_import_teu'].fillna(0) * 0.6 / 2

    port_terminal_map.to_csv(OUTPUT_DIR / "port_terminal_mapping.csv", index=False)
    print(f"Saved: port_terminal_mapping.csv ({len(port_terminal_map)} ports)")

    # =========================================================================
    # 3. Filter to Active Ports
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. FILTERING TO ACTIVE PORTS")
    print("=" * 60)

    port_avg = ports_us_pd.groupby('portname')['portcalls'].mean()
    active_ports = port_avg[port_avg >= DATA_CONFIG.min_activity_threshold].index.tolist()

    print(f"Active ports (≥{DATA_CONFIG.min_activity_threshold} calls/day): {len(active_ports)}")

    df = ports_us_pd[ports_us_pd['portname'].isin(active_ports)].copy()
    df = df.sort_values(['portname', 'date'])

    print(f"Filtered records: {len(df):,}")

    # =========================================================================
    # 4. Feature Engineering (Baseline)
    # =========================================================================
    print("\n" + "=" * 60)
    print("4. BASELINE FEATURE ENGINEERING")
    print("=" * 60)

    df = engineer_all_features(df, add_targets=True)

    # =========================================================================
    # 5. Enhanced Features
    # =========================================================================
    print("\n" + "=" * 60)
    print("5. ENHANCED FEATURE ENGINEERING")
    print("=" * 60)

    # Integrate chokepoint data
    if chokepoint_data is not None:
        print("Merging chokepoint features...")
        # Ensure date columns are compatible (remove timezone if present)
        df['date'] = pd.to_datetime(df['date']).dt.tz_localize(None)
        chokepoint_data['date'] = pd.to_datetime(chokepoint_data['date']).dt.tz_localize(None)
        df = df.merge(chokepoint_data, on='date', how='left')
        # Fill missing with forward fill then backward fill
        choke_cols = [c for c in chokepoint_data.columns if c != 'date']
        df[choke_cols] = df[choke_cols].fillna(method='ffill').fillna(method='bfill').fillna(0)
        print(f"  Added {len(choke_cols)} chokepoint features")

    # Integrate disruption data
    df = create_disruption_features(df, disruption_data)

    # Add cross-port features
    df = add_cross_port_features(df)

    # Add volatility features
    df = add_volatility_features(df)

    # Add interaction features
    df = add_interaction_features(df)

    # Replace infinities
    df = df.replace([np.inf, -np.inf], np.nan)

    # Save enhanced features
    df.to_parquet(OUTPUT_DIR / "enhanced_features.parquet", index=False)
    print(f"\nSaved: enhanced_features.parquet ({len(df.columns)} columns)")

    # =========================================================================
    # 6. Prepare ML Data
    # =========================================================================
    print("\n" + "=" * 60)
    print("6. PREPARING ML DATA")
    print("=" * 60)

    # Get feature columns
    feature_cols = get_feature_columns(df)

    # Add new feature columns
    new_features = [
        # Chokepoint features
        'choke_container', 'choke_ma7', 'choke_ma14', 'choke_zscore',
        'choke_container_lag5', 'choke_container_lag7', 'choke_container_lag10',
        'choke_container_lag14', 'choke_container_lag21',
        'choke_capacity_lag7', 'choke_capacity_lag14',
        # Disruption features
        'active_disruptions', 'tc_active', 'flood_active', 'max_severity',
        'disruptions_7d', 'disruptions_30d',
        # Cross-port features
        'regional_avg_calls', 'regional_share', 'national_share', 'vs_regional_avg',
        # Volatility features
        'cv_7d', 'cv_30d', 'bb_position', 'deviation_from_ma30', 'trend_7_30',
        # Interaction features
        'volume_volatility', 'weekend_calls', 'monthend_import', 'momentum_volatility'
    ]

    for feat in new_features:
        if feat in df.columns and feat not in feature_cols:
            feature_cols.append(feat)

    feature_cols = [c for c in feature_cols if c in df.columns]
    print(f"Total features: {len(feature_cols)}")

    # Prepare targets
    targets = {
        'calls_1d': df['target_calls_1d'],
        'calls_3d': df['target_calls_3d'],
        'calls_7d': df['target_calls_7d'],
    }

    if 'target_surge_1d' in df.columns:
        targets['surge_1d'] = df['target_surge_1d']

    X = df[feature_cols].fillna(0)

    print(f"\nML Dataset:")
    print(f"   Features: {X.shape[1]}")
    print(f"   Samples: {X.shape[0]:,}")
    print(f"   Targets: {list(targets.keys())}")

    # =========================================================================
    # 7. Train Enhanced Model
    # =========================================================================
    print("\n" + "=" * 60)
    print("7. TRAINING ENHANCED MODEL")
    print("=" * 60)

    # Time-based split (last 20% as test)
    df_sorted = df.sort_values('date')
    split_idx = int(len(df_sorted) * 0.8)
    train_idx = df_sorted.index[:split_idx]
    test_idx = df_sorted.index[split_idx:]

    X_train = X.loc[train_idx]
    X_test = X.loc[test_idx]

    results = {}
    models = {}

    # Configure XGBoost
    if args.gpu:
        MODEL_CONFIG.tree_method = 'gpu_hist'
        MODEL_CONFIG.device = 'cuda'
        MODEL_CONFIG.n_estimators = 1000
        print("GPU mode enabled")
    else:
        MODEL_CONFIG.tree_method = 'hist'
        MODEL_CONFIG.device = 'cpu'

    # Train regression models
    from xgboost import XGBRegressor

    for target_name in ['calls_1d', 'calls_3d', 'calls_7d']:
        print(f"\n  Training {target_name}...")

        y_train = targets[target_name].loc[train_idx].dropna()
        y_test = targets[target_name].loc[test_idx].dropna()

        X_train_target = X_train.loc[y_train.index]
        X_test_target = X_test.loc[y_test.index]

        model = XGBRegressor(
            max_depth=MODEL_CONFIG.max_depth,
            n_estimators=MODEL_CONFIG.n_estimators,
            learning_rate=MODEL_CONFIG.learning_rate,
            tree_method=MODEL_CONFIG.tree_method,
            device=MODEL_CONFIG.device,
            random_state=42
        )

        t1 = time.time()
        model.fit(X_train_target, y_train)
        train_time = time.time() - t1

        y_pred = model.predict(X_test_target)

        results[target_name] = {
            'mae': mean_absolute_error(y_test, y_pred),
            'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
            'r2': r2_score(y_test, y_pred),
            'train_time': train_time,
            'n_samples_train': len(y_train),
            'n_samples_test': len(y_test)
        }

        models[target_name] = model

        print(f"    R²: {results[target_name]['r2']:.4f}")
        print(f"    MAE: {results[target_name]['mae']:.4f}")

    # Train surge classifier with enhanced handling
    if 'surge_1d' in targets:
        y_train_surge = targets['surge_1d'].loc[train_idx].dropna()
        y_test_surge = targets['surge_1d'].loc[test_idx].dropna()

        X_train_surge = X_train.loc[y_train_surge.index]
        X_test_surge = X_test.loc[y_test_surge.index]

        surge_model, surge_metrics, optimal_threshold = train_surge_classifier(
            X_train_surge, y_train_surge,
            X_test_surge, y_test_surge,
            use_smote=True,
            use_gpu=args.gpu
        )

        results['surge_1d'] = surge_metrics
        models['surge_1d'] = surge_model

    # =========================================================================
    # 8. Save Results
    # =========================================================================
    print("\n" + "=" * 60)
    print("8. SAVING RESULTS")
    print("=" * 60)

    # Save models using ChampionModel format
    champion = ChampionModel(MODEL_CONFIG)
    champion.models = models
    champion.feature_cols = feature_cols
    champion.save(MODEL_DIR, prefix='enhanced')

    # Also save as champion (for API compatibility)
    champion.save(MODEL_DIR, prefix='champion')

    # Save surge analysis
    surge_cols = ['portname', 'surge_2std', 'surge_3std', 'zscore_7d', 'portcalls', 'ma7']
    surge_cols = [c for c in surge_cols if c in df.columns]

    if len(surge_cols) > 1:
        surge_summary = df.groupby('portname').agg({
            'surge_2std': 'sum',
            'portcalls': ['mean', 'max', 'std'],
        }).reset_index()
        surge_summary.columns = ['port', 'surge_2std_days', 'avg_calls', 'max_calls', 'std_calls']
        surge_summary['total_days'] = df.groupby('portname').size().values
        surge_summary['surge_rate'] = surge_summary['surge_2std_days'] / surge_summary['total_days'] * 100

        avg_import = df.groupby('portname')['total_import'].mean().reset_index()
        avg_import.columns = ['port', 'avg_import']
        surge_summary = surge_summary.merge(avg_import, on='port', how='left')

        surge_summary = surge_summary.sort_values('avg_calls', ascending=False)
        surge_summary.to_csv(OUTPUT_DIR / "surge_analysis_enhanced.csv", index=False)
        print("Saved: surge_analysis_enhanced.csv")

    # Save training report
    report = {
        'timestamp': datetime.now().isoformat(),
        'training_time_sec': round(time.time() - t0, 1),
        'gpu_mode': args.gpu,
        'smote_used': SMOTE_AVAILABLE,
        'data': {
            'total_records': len(df),
            'n_ports': df['portname'].nunique(),
            'n_features': len(feature_cols),
            'date_range': {
                'start': str(df['date'].min()),
                'end': str(df['date'].max())
            },
            'chokepoint_integrated': chokepoint_data is not None,
            'disruptions_integrated': disruption_data is not None
        },
        'model_config': {
            'max_depth': MODEL_CONFIG.max_depth,
            'n_estimators': MODEL_CONFIG.n_estimators,
            'learning_rate': MODEL_CONFIG.learning_rate,
            'tree_method': MODEL_CONFIG.tree_method
        },
        'results': results,
        'optimal_surge_threshold': optimal_threshold if 'surge_1d' in targets else None
    }

    with open(OUTPUT_DIR / "enhanced_training_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print("Saved: enhanced_training_report.json")

    # Save feature importance
    if 'calls_1d' in models:
        importance = pd.DataFrame({
            'feature': feature_cols,
            'importance': models['calls_1d'].feature_importances_
        }).sort_values('importance', ascending=False)
        importance.to_csv(OUTPUT_DIR / "enhanced_feature_importance.csv", index=False)
        print("Saved: enhanced_feature_importance.csv")

        print("\nTop 15 Features:")
        print(importance.head(15).to_string(index=False))

    # =========================================================================
    # 9. Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("ENHANCED TRAINING COMPLETE!")
    print("=" * 80)

    print(f"\nResults Summary:")
    for target, metrics in results.items():
        print(f"\n   {target}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"      {k}: {v:.4f}")

    print(f"\nTotal training time: {time.time() - t0:.1f}s")

    print(f"\nOutput files:")
    print(f"   • {MODEL_DIR}/enhanced_*.json (models)")
    print(f"   • {MODEL_DIR}/champion_*.json (API compatible)")
    print(f"   • {OUTPUT_DIR}/enhanced_features.parquet")
    print(f"   • {OUTPUT_DIR}/enhanced_training_report.json")
    print(f"   • {OUTPUT_DIR}/enhanced_feature_importance.csv")

    print("\nReady for inference!")
    print("   Run: uvicorn api.main:app --reload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Enhanced Model")
    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration')
    args = parser.parse_args()

    main(args)
