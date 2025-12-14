#!/usr/bin/env python3
"""
Train the Champion Model for Port-to-Rail Surge Forecaster.

This script:
1. Loads all port activity data
2. Engineers 64+ features
3. Trains XGBoost models for multiple horizons
4. Saves models and generates reports

Usage:
    python train_champion_model.py

For GPU (DGX):
    python train_champion_model.py --gpu
"""

import argparse
import json
import time
from datetime import datetime
from pathlib import Path

import numpy as np
import pandas as pd

# Add src to path
import sys
sys.path.insert(0, str(Path(__file__).parent))

from src.config import DATA_DIR, OUTPUT_DIR, MODEL_DIR, MODEL_CONFIG, DATA_CONFIG
from src.data_loader import (
    load_port_activity, load_port_database, load_rail_nodes,
    filter_us_data, parse_dates, create_port_terminal_mapping
)
from src.feature_engineering import (
    engineer_all_features, prepare_ml_data, get_feature_columns
)
from src.model import ChampionModel, train_champion_model


def main(args):
    """Main training pipeline."""

    print("=" * 80)
    print("🚢 PORT-TO-RAIL SURGE FORECASTER - CHAMPION MODEL TRAINING")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")
    print(f"GPU mode: {args.gpu}")
    print()

    t0 = time.time()

    # =========================================================================
    # 1. Load Data
    # =========================================================================
    print("\n" + "=" * 60)
    print("1. LOADING DATA")
    print("=" * 60)

    try:
        port_activity = load_port_activity(DATA_DIR)
        port_db = load_port_database(DATA_DIR)
        rail_nodes = load_rail_nodes(DATA_DIR)
    except FileNotFoundError as e:
        print(f"❌ Error loading data: {e}")
        print("Make sure data files are in the 'data/' directory")
        return

    # Filter to US
    ports_us, ports_db_us, rail_us = filter_us_data(port_activity, port_db, rail_nodes)

    # Parse dates
    ports_us = parse_dates(ports_us)

    # Free memory
    del port_activity, port_db, rail_nodes
    import gc
    gc.collect()

    print(f"\n📊 Data Summary:")
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

    # Save
    port_terminal_map.to_csv(OUTPUT_DIR / "port_terminal_mapping.csv", index=False)
    print(f"\n✅ Saved: port_terminal_mapping.csv ({len(port_terminal_map)} ports)")

    # =========================================================================
    # 3. Filter to Active Ports
    # =========================================================================
    print("\n" + "=" * 60)
    print("3. FILTERING TO ACTIVE PORTS")
    print("=" * 60)

    # Filter to ports with enough activity
    port_avg = ports_us_pd.groupby('portname')['portcalls'].mean()
    active_ports = port_avg[port_avg >= DATA_CONFIG.min_activity_threshold].index.tolist()

    print(f"Active ports (≥{DATA_CONFIG.min_activity_threshold} calls/day): {len(active_ports)}")

    df = ports_us_pd[ports_us_pd['portname'].isin(active_ports)].copy()
    df = df.sort_values(['portname', 'date'])

    print(f"Filtered records: {len(df):,}")

    # =========================================================================
    # 4. Feature Engineering
    # =========================================================================
    print("\n" + "=" * 60)
    print("4. FEATURE ENGINEERING")
    print("=" * 60)

    df = engineer_all_features(df, add_targets=True)

    # Save features
    df.to_parquet(OUTPUT_DIR / "champion_features.parquet", index=False)
    print(f"\n✅ Saved: champion_features.parquet ({len(df.columns)} columns)")

    # =========================================================================
    # 5. Prepare ML Data
    # =========================================================================
    print("\n" + "=" * 60)
    print("5. PREPARING ML DATA")
    print("=" * 60)

    # Get feature columns
    feature_cols = get_feature_columns(df)
    print(f"Features available: {len(feature_cols)}")

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
    # 6. Train Champion Model
    # =========================================================================
    print("\n" + "=" * 60)
    print("6. TRAINING CHAMPION MODEL")
    print("=" * 60)

    # Update config for GPU if specified
    if args.gpu:
        MODEL_CONFIG.tree_method = 'gpu_hist'
        MODEL_CONFIG.device = 'cuda'
        MODEL_CONFIG.max_bin = 1024  # Higher for GPU
        MODEL_CONFIG.n_estimators = 1000  # More trees with GPU
        print("🚀 GPU mode enabled")
    else:
        MODEL_CONFIG.tree_method = 'hist'
        MODEL_CONFIG.device = 'cpu'

    model = ChampionModel(MODEL_CONFIG)
    results = model.train(X, targets, test_size=0.2)

    # =========================================================================
    # 7. Save Results
    # =========================================================================
    print("\n" + "=" * 60)
    print("7. SAVING RESULTS")
    print("=" * 60)

    # Save model
    model.save(MODEL_DIR, prefix='champion')

    # Save surge analysis
    surge_cols = ['portname', 'surge_2std', 'surge_3std', 'zscore_7d', 'portcalls', 'ma7']
    surge_cols = [c for c in surge_cols if c in df.columns]

    if len(surge_cols) > 1:
        surge_summary = df.groupby('portname').agg({
            'surge_2std': 'sum',
            'portcalls': ['mean', 'max', 'std'],
            'zscore_7d': 'max' if 'zscore_7d' in df.columns else 'portcalls'
        }).reset_index()
        surge_summary.columns = ['port', 'surge_2std_days', 'avg_calls', 'max_calls', 'std_calls', 'max_zscore']
        surge_summary['total_days'] = df.groupby('portname').size().values
        surge_summary['surge_rate'] = surge_summary['surge_2std_days'] / surge_summary['total_days'] * 100

        # Add avg import
        avg_import = df.groupby('portname')['total_import'].mean().reset_index()
        avg_import.columns = ['port', 'avg_import']
        surge_summary = surge_summary.merge(avg_import, on='port', how='left')

        surge_summary = surge_summary.sort_values('avg_calls', ascending=False)
        surge_summary.to_csv(OUTPUT_DIR / "surge_analysis.csv", index=False)
        print(f"✅ Saved: surge_analysis.csv")

    # Save training report
    report = {
        'timestamp': datetime.now().isoformat(),
        'training_time_sec': round(time.time() - t0, 1),
        'gpu_mode': args.gpu,
        'data': {
            'total_records': len(df),
            'n_ports': df['portname'].nunique(),
            'n_features': len(feature_cols),
            'date_range': {
                'start': str(df['date'].min()),
                'end': str(df['date'].max())
            }
        },
        'model_config': {
            'max_depth': MODEL_CONFIG.max_depth,
            'n_estimators': MODEL_CONFIG.n_estimators,
            'learning_rate': MODEL_CONFIG.learning_rate,
            'tree_method': MODEL_CONFIG.tree_method
        },
        'results': results
    }

    with open(OUTPUT_DIR / "training_report.json", 'w') as f:
        json.dump(report, f, indent=2, default=str)
    print(f"✅ Saved: training_report.json")

    # =========================================================================
    # 8. Summary
    # =========================================================================
    print("\n" + "=" * 80)
    print("🎉 TRAINING COMPLETE!")
    print("=" * 80)

    print(f"\n📊 Results Summary:")
    for target, metrics in results.items():
        print(f"\n   {target}:")
        for k, v in metrics.items():
            if isinstance(v, float):
                print(f"      {k}: {v:.4f}")

    print(f"\n⏱️ Total training time: {time.time() - t0:.1f}s")

    print(f"\n📁 Output files:")
    print(f"   • {MODEL_DIR}/champion_*.json (models)")
    print(f"   • {OUTPUT_DIR}/champion_features.parquet")
    print(f"   • {OUTPUT_DIR}/port_terminal_mapping.csv")
    print(f"   • {OUTPUT_DIR}/surge_analysis.csv")
    print(f"   • {OUTPUT_DIR}/training_report.json")

    print("\n✅ Ready for inference!")
    print("   Run: uvicorn api.main:app --reload")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="Train Champion Model")
    parser.add_argument('--gpu', action='store_true', help='Enable GPU acceleration')
    args = parser.parse_args()

    main(args)
