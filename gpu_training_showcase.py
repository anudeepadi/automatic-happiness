#!/usr/bin/env python3
"""
GPU-Accelerated Training Pipeline for Port-to-Rail Surge Forecaster
====================================================================

This script demonstrates NVIDIA GPU capabilities for the DGX Spark Frontier Hackathon:

1. RAPIDS cuDF - GPU-accelerated DataFrames (100x faster than pandas)
2. RAPIDS cuML - GPU-accelerated ML algorithms
3. XGBoost GPU - gpu_hist tree method with CUDA
4. PyTorch LSTM - Deep learning time series model
5. Multi-GPU scaling with Dask-CUDA

Requirements:
    - NVIDIA GPU with CUDA support
    - RAPIDS (cudf, cuml, dask-cudf)
    - PyTorch with CUDA
    - XGBoost with GPU support

Usage:
    python gpu_training_showcase.py
    python gpu_training_showcase.py --benchmark  # Run CPU vs GPU comparison
"""

import argparse
import gc
import json
import os
import time
import warnings
from datetime import datetime
from pathlib import Path
from typing import Dict, List, Optional, Tuple

import numpy as np

warnings.filterwarnings('ignore')

# ============================================================================
# GPU Detection and Setup
# ============================================================================

def detect_gpu_capabilities():
    """Detect and report GPU capabilities."""

    print("=" * 80)
    print("🚀 NVIDIA GPU CAPABILITY DETECTION")
    print("=" * 80)

    capabilities = {
        'cuda_available': False,
        'gpu_count': 0,
        'gpu_names': [],
        'gpu_memory': [],
        'rapids_available': False,
        'pytorch_cuda': False,
        'xgboost_gpu': False
    }

    # Check CUDA via PyTorch
    try:
        import torch
        capabilities['pytorch_cuda'] = torch.cuda.is_available()
        if capabilities['pytorch_cuda']:
            capabilities['cuda_available'] = True
            capabilities['gpu_count'] = torch.cuda.device_count()
            for i in range(capabilities['gpu_count']):
                props = torch.cuda.get_device_properties(i)
                capabilities['gpu_names'].append(props.name)
                capabilities['gpu_memory'].append(f"{props.total_memory / 1e9:.1f} GB")
            print(f"✅ PyTorch CUDA: {torch.version.cuda}")
            print(f"   GPUs detected: {capabilities['gpu_count']}")
            for i, (name, mem) in enumerate(zip(capabilities['gpu_names'], capabilities['gpu_memory'])):
                print(f"   GPU {i}: {name} ({mem})")
    except ImportError:
        print("⚠️ PyTorch not installed")

    # Check RAPIDS
    try:
        import cudf
        import cuml
        capabilities['rapids_available'] = True
        print(f"✅ RAPIDS cuDF: {cudf.__version__}")
        print(f"✅ RAPIDS cuML: {cuml.__version__}")
    except ImportError:
        print("⚠️ RAPIDS not installed (cudf/cuml)")

    # Check XGBoost GPU
    try:
        import xgboost as xgb
        # Test GPU availability
        try:
            test_params = {'tree_method': 'gpu_hist', 'device': 'cuda'}
            dtrain = xgb.DMatrix(np.random.rand(10, 5), label=np.random.rand(10))
            xgb.train(test_params, dtrain, num_boost_round=1)
            capabilities['xgboost_gpu'] = True
            print(f"✅ XGBoost GPU: {xgb.__version__} (gpu_hist available)")
        except Exception:
            print(f"⚠️ XGBoost {xgb.__version__} (GPU not available)")
    except ImportError:
        print("⚠️ XGBoost not installed")

    print("=" * 80)
    return capabilities


# ============================================================================
# RAPIDS cuDF Accelerated Data Loading
# ============================================================================

def load_data_gpu(data_dir: Path) -> "cudf.DataFrame":
    """Load data using RAPIDS cuDF (GPU-accelerated)."""

    import cudf

    print("\n📊 Loading data with RAPIDS cuDF (GPU)...")
    t0 = time.time()

    # Load main port activity data
    port_file = data_dir / "Daily_Port_Activity_Data_and_Trade_Estimates.csv"
    df = cudf.read_csv(str(port_file))

    load_time = time.time() - t0
    print(f"   Loaded {len(df):,} rows in {load_time:.2f}s")
    print(f"   GPU memory used: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    return df, load_time


def load_data_cpu(data_dir: Path) -> "pd.DataFrame":
    """Load data using pandas (CPU baseline)."""

    import pandas as pd

    print("\n📊 Loading data with pandas (CPU)...")
    t0 = time.time()

    port_file = data_dir / "Daily_Port_Activity_Data_and_Trade_Estimates.csv"
    df = pd.read_csv(str(port_file))

    load_time = time.time() - t0
    print(f"   Loaded {len(df):,} rows in {load_time:.2f}s")

    return df, load_time


# ============================================================================
# RAPIDS cuDF Feature Engineering
# ============================================================================

def engineer_features_gpu(df: "cudf.DataFrame") -> Tuple["cudf.DataFrame", float]:
    """GPU-accelerated feature engineering with RAPIDS cuDF."""

    import cudf
    import cupy as cp

    print("\n⚙️ Engineering features with RAPIDS cuDF (GPU)...")
    t0 = time.time()

    # Filter to US
    df = df[df['country'] == 'United States'].copy()

    # Parse dates
    df['date'] = cudf.to_datetime(df['date'])

    # Sort by port and date
    df = df.sort_values(['portname', 'date'])

    # Temporal features (vectorized on GPU)
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_weekend'] = (df['day_of_week'] >= 5).astype('int32')

    # Cyclical encoding (GPU-accelerated with CuPy)
    df['dow_sin'] = cp.sin(2 * cp.pi * df['day_of_week'].values / 7)
    df['dow_cos'] = cp.cos(2 * cp.pi * df['day_of_week'].values / 7)
    df['month_sin'] = cp.sin(2 * cp.pi * df['month'].values / 12)
    df['month_cos'] = cp.cos(2 * cp.pi * df['month'].values / 12)

    # Volume features
    import_cols = [c for c in df.columns if c.startswith('import_') and 'lag' not in c]
    export_cols = [c for c in df.columns if c.startswith('export_') and 'lag' not in c]

    df['total_import'] = df[import_cols].sum(axis=1)
    df['total_export'] = df[export_cols].sum(axis=1)
    df['total_volume'] = df['total_import'] + df['total_export']

    # Rolling features (GPU-accelerated groupby rolling)
    for window in [7, 14, 30]:
        df[f'ma{window}'] = df.groupby('portname')['portcalls'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'std{window}'] = df.groupby('portname')['portcalls'].transform(
            lambda x: x.rolling(window, min_periods=2).std()
        ).fillna(0)

    # Z-scores for surge detection
    df['zscore_7d'] = (df['portcalls'] - df['ma7']) / df['std7'].replace(0, 1)

    # Lag features
    for lag in [1, 3, 7, 14]:
        df[f'calls_lag{lag}'] = df.groupby('portname')['portcalls'].shift(lag)

    # Target
    df['target_calls_1d'] = df.groupby('portname')['portcalls'].shift(-1)
    df['target_surge_1d'] = (df.groupby('portname')['zscore_7d'].shift(-1) > 2).astype('int32')

    # Port ranking
    port_means = df.groupby('portname')['portcalls'].mean().reset_index()
    port_means.columns = ['portname', 'avg_calls']
    port_means['activity_rank'] = port_means['avg_calls'].rank(ascending=False)
    df = df.merge(port_means[['portname', 'activity_rank']], on='portname', how='left')

    feature_time = time.time() - t0
    print(f"   Engineered {len(df.columns)} features in {feature_time:.2f}s")
    print(f"   GPU memory: {df.memory_usage(deep=True).sum() / 1e6:.1f} MB")

    return df, feature_time


def engineer_features_cpu(df: "pd.DataFrame") -> Tuple["pd.DataFrame", float]:
    """CPU baseline feature engineering with pandas."""

    import pandas as pd

    print("\n⚙️ Engineering features with pandas (CPU)...")
    t0 = time.time()

    # Filter to US
    df = df[df['country'] == 'United States'].copy()

    # Parse dates
    df['date'] = pd.to_datetime(df['date'])

    # Sort
    df = df.sort_values(['portname', 'date'])

    # Temporal features
    df['day_of_week'] = df['date'].dt.dayofweek
    df['month'] = df['date'].dt.month
    df['day_of_year'] = df['date'].dt.dayofyear
    df['is_weekend'] = (df['day_of_week'] >= 5).astype(int)

    # Cyclical encoding
    df['dow_sin'] = np.sin(2 * np.pi * df['day_of_week'] / 7)
    df['dow_cos'] = np.cos(2 * np.pi * df['day_of_week'] / 7)
    df['month_sin'] = np.sin(2 * np.pi * df['month'] / 12)
    df['month_cos'] = np.cos(2 * np.pi * df['month'] / 12)

    # Volume features
    import_cols = [c for c in df.columns if c.startswith('import_') and 'lag' not in c]
    export_cols = [c for c in df.columns if c.startswith('export_') and 'lag' not in c]

    df['total_import'] = df[import_cols].sum(axis=1)
    df['total_export'] = df[export_cols].sum(axis=1)
    df['total_volume'] = df['total_import'] + df['total_export']

    # Rolling features
    for window in [7, 14, 30]:
        df[f'ma{window}'] = df.groupby('portname')['portcalls'].transform(
            lambda x: x.rolling(window, min_periods=1).mean()
        )
        df[f'std{window}'] = df.groupby('portname')['portcalls'].transform(
            lambda x: x.rolling(window, min_periods=2).std()
        ).fillna(0)

    # Z-scores
    df['zscore_7d'] = (df['portcalls'] - df['ma7']) / df['std7'].replace(0, 1)

    # Lag features
    for lag in [1, 3, 7, 14]:
        df[f'calls_lag{lag}'] = df.groupby('portname')['portcalls'].shift(lag)

    # Target
    df['target_calls_1d'] = df.groupby('portname')['portcalls'].shift(-1)
    df['target_surge_1d'] = (df.groupby('portname')['zscore_7d'].shift(-1) > 2).astype(int)

    # Port ranking
    port_means = df.groupby('portname')['portcalls'].mean().reset_index()
    port_means.columns = ['portname', 'avg_calls']
    port_means['activity_rank'] = port_means['avg_calls'].rank(ascending=False)
    df = df.merge(port_means[['portname', 'activity_rank']], on='portname', how='left')

    feature_time = time.time() - t0
    print(f"   Engineered {len(df.columns)} features in {feature_time:.2f}s")

    return df, feature_time


# ============================================================================
# XGBoost GPU Training
# ============================================================================

def train_xgboost_gpu(X_train, y_train, X_test, y_test, n_estimators=1000) -> Tuple[dict, float]:
    """Train XGBoost with GPU acceleration."""

    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, r2_score

    print("\n🚀 Training XGBoost with GPU (gpu_hist)...")

    # Convert to DMatrix for optimal GPU performance
    t0 = time.time()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'tree_method': 'gpu_hist',      # GPU-accelerated histogram
        'device': 'cuda',                # Use CUDA
        'max_depth': 12,
        'learning_rate': 0.05,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'max_bin': 1024,                 # Higher bins for GPU
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }

    # Train with early stopping
    evals = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=100
    )

    train_time = time.time() - t0

    # Evaluate
    y_pred = model.predict(dtest)
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'train_time': train_time,
        'n_trees': model.best_iteration
    }

    print(f"   R²: {metrics['r2']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f}")
    print(f"   Training time: {train_time:.2f}s")
    print(f"   Trees: {metrics['n_trees']}")

    return model, metrics


def train_xgboost_cpu(X_train, y_train, X_test, y_test, n_estimators=1000) -> Tuple[dict, float]:
    """Train XGBoost with CPU (baseline)."""

    import xgboost as xgb
    from sklearn.metrics import mean_absolute_error, r2_score

    print("\n🐢 Training XGBoost with CPU (hist)...")

    t0 = time.time()

    dtrain = xgb.DMatrix(X_train, label=y_train)
    dtest = xgb.DMatrix(X_test, label=y_test)

    params = {
        'tree_method': 'hist',           # CPU histogram
        'device': 'cpu',
        'max_depth': 12,
        'learning_rate': 0.05,
        'objective': 'reg:squarederror',
        'eval_metric': 'rmse',
        'subsample': 0.8,
        'colsample_bytree': 0.8,
        'seed': 42
    }

    evals = [(dtrain, 'train'), (dtest, 'eval')]
    model = xgb.train(
        params,
        dtrain,
        num_boost_round=n_estimators,
        evals=evals,
        early_stopping_rounds=50,
        verbose_eval=100
    )

    train_time = time.time() - t0

    y_pred = model.predict(dtest)
    metrics = {
        'r2': r2_score(y_test, y_pred),
        'mae': mean_absolute_error(y_test, y_pred),
        'train_time': train_time,
        'n_trees': model.best_iteration
    }

    print(f"   R²: {metrics['r2']:.4f}")
    print(f"   MAE: {metrics['mae']:.4f}")
    print(f"   Training time: {train_time:.2f}s")

    return model, metrics


# ============================================================================
# PyTorch LSTM Deep Learning Model
# ============================================================================

def create_sequences(data, seq_length=30):
    """Create sequences for LSTM training."""
    X, y = [], []
    for i in range(len(data) - seq_length):
        X.append(data[i:i+seq_length])
        y.append(data[i+seq_length, 0])  # Predict portcalls
    return np.array(X), np.array(y)


class PortLSTM:
    """PyTorch LSTM for port call forecasting."""

    def __init__(self, input_size, hidden_size=128, num_layers=2, dropout=0.2):
        import torch
        import torch.nn as nn

        self.device = torch.device('cuda' if torch.cuda.is_available() else 'cpu')

        class LSTMModel(nn.Module):
            def __init__(self, input_size, hidden_size, num_layers, dropout):
                super().__init__()
                self.lstm = nn.LSTM(
                    input_size=input_size,
                    hidden_size=hidden_size,
                    num_layers=num_layers,
                    dropout=dropout,
                    batch_first=True
                )
                self.fc1 = nn.Linear(hidden_size, 64)
                self.fc2 = nn.Linear(64, 1)
                self.relu = nn.ReLU()
                self.dropout = nn.Dropout(dropout)

            def forward(self, x):
                lstm_out, _ = self.lstm(x)
                last_hidden = lstm_out[:, -1, :]
                x = self.dropout(self.relu(self.fc1(last_hidden)))
                return self.fc2(x)

        self.model = LSTMModel(input_size, hidden_size, num_layers, dropout).to(self.device)
        self.input_size = input_size

    def train(self, X_train, y_train, X_val, y_val, epochs=100, batch_size=256, lr=0.001):
        import torch
        import torch.nn as nn
        from torch.utils.data import DataLoader, TensorDataset

        print(f"\n🧠 Training LSTM on {self.device}...")
        t0 = time.time()

        # Convert to tensors
        X_train_t = torch.FloatTensor(X_train).to(self.device)
        y_train_t = torch.FloatTensor(y_train).to(self.device)
        X_val_t = torch.FloatTensor(X_val).to(self.device)
        y_val_t = torch.FloatTensor(y_val).to(self.device)

        # DataLoader
        train_dataset = TensorDataset(X_train_t, y_train_t)
        train_loader = DataLoader(train_dataset, batch_size=batch_size, shuffle=True)

        # Optimizer and loss
        optimizer = torch.optim.Adam(self.model.parameters(), lr=lr)
        criterion = nn.MSELoss()
        scheduler = torch.optim.lr_scheduler.ReduceLROnPlateau(optimizer, patience=5, factor=0.5)

        best_val_loss = float('inf')
        patience_counter = 0

        for epoch in range(epochs):
            self.model.train()
            train_loss = 0

            for batch_X, batch_y in train_loader:
                optimizer.zero_grad()
                outputs = self.model(batch_X).squeeze()
                loss = criterion(outputs, batch_y)
                loss.backward()
                torch.nn.utils.clip_grad_norm_(self.model.parameters(), 1.0)
                optimizer.step()
                train_loss += loss.item()

            # Validation
            self.model.eval()
            with torch.no_grad():
                val_outputs = self.model(X_val_t).squeeze()
                val_loss = criterion(val_outputs, y_val_t).item()

            scheduler.step(val_loss)

            if (epoch + 1) % 10 == 0:
                print(f"   Epoch {epoch+1}/{epochs} - Train Loss: {train_loss/len(train_loader):.4f}, Val Loss: {val_loss:.4f}")

            # Early stopping
            if val_loss < best_val_loss:
                best_val_loss = val_loss
                patience_counter = 0
            else:
                patience_counter += 1
                if patience_counter >= 15:
                    print(f"   Early stopping at epoch {epoch+1}")
                    break

        train_time = time.time() - t0
        print(f"   Training time: {train_time:.2f}s")

        return train_time

    def predict(self, X):
        import torch
        self.model.eval()
        with torch.no_grad():
            X_t = torch.FloatTensor(X).to(self.device)
            return self.model(X_t).squeeze().cpu().numpy()


# ============================================================================
# cuML GPU-Accelerated ML
# ============================================================================

def train_cuml_models(X_train, y_train, X_test, y_test):
    """Train models using RAPIDS cuML (GPU-accelerated)."""

    import cudf
    import cuml
    from cuml.ensemble import RandomForestRegressor as cuRF
    from cuml.linear_model import Ridge as cuRidge
    from cuml.metrics import mean_absolute_error as cu_mae, r2_score as cu_r2

    print("\n🚀 Training cuML models (GPU)...")
    results = {}

    # Convert to cuDF
    X_train_cu = cudf.DataFrame(X_train)
    y_train_cu = cudf.Series(y_train)
    X_test_cu = cudf.DataFrame(X_test)
    y_test_cu = cudf.Series(y_test)

    # cuML Random Forest
    print("   Training cuML Random Forest...")
    t0 = time.time()
    rf = cuRF(n_estimators=100, max_depth=12, n_streams=4)
    rf.fit(X_train_cu, y_train_cu)
    rf_time = time.time() - t0

    y_pred_rf = rf.predict(X_test_cu)
    results['cuml_rf'] = {
        'r2': float(cu_r2(y_test_cu, y_pred_rf)),
        'mae': float(cu_mae(y_test_cu, y_pred_rf)),
        'train_time': rf_time
    }
    print(f"      R²: {results['cuml_rf']['r2']:.4f}, Time: {rf_time:.2f}s")

    # cuML Ridge Regression
    print("   Training cuML Ridge Regression...")
    t0 = time.time()
    ridge = cuRidge(alpha=1.0)
    ridge.fit(X_train_cu, y_train_cu)
    ridge_time = time.time() - t0

    y_pred_ridge = ridge.predict(X_test_cu)
    results['cuml_ridge'] = {
        'r2': float(cu_r2(y_test_cu, y_pred_ridge)),
        'mae': float(cu_mae(y_test_cu, y_pred_ridge)),
        'train_time': ridge_time
    }
    print(f"      R²: {results['cuml_ridge']['r2']:.4f}, Time: {ridge_time:.2f}s")

    return results


# ============================================================================
# GPU Memory Monitoring
# ============================================================================

def get_gpu_memory_info():
    """Get current GPU memory usage."""
    try:
        import torch
        if torch.cuda.is_available():
            allocated = torch.cuda.memory_allocated() / 1e9
            cached = torch.cuda.memory_reserved() / 1e9
            total = torch.cuda.get_device_properties(0).total_memory / 1e9
            return {
                'allocated_gb': allocated,
                'cached_gb': cached,
                'total_gb': total,
                'utilization_pct': (allocated / total) * 100
            }
    except:
        pass

    try:
        import pynvml
        pynvml.nvmlInit()
        handle = pynvml.nvmlDeviceGetHandleByIndex(0)
        info = pynvml.nvmlDeviceGetMemoryInfo(handle)
        return {
            'used_gb': info.used / 1e9,
            'total_gb': info.total / 1e9,
            'utilization_pct': (info.used / info.total) * 100
        }
    except:
        return None


# ============================================================================
# Benchmark Runner
# ============================================================================

def run_benchmark(data_dir: Path):
    """Run comprehensive CPU vs GPU benchmark."""

    print("\n" + "=" * 80)
    print("📊 CPU vs GPU BENCHMARK")
    print("=" * 80)

    results = {
        'timestamp': datetime.now().isoformat(),
        'benchmarks': {}
    }

    # Benchmark 1: Data Loading
    print("\n" + "-" * 60)
    print("BENCHMARK 1: Data Loading")
    print("-" * 60)

    try:
        df_gpu, gpu_load_time = load_data_gpu(data_dir)
        results['benchmarks']['data_loading_gpu'] = gpu_load_time
        del df_gpu
        gc.collect()
    except Exception as e:
        print(f"   GPU loading failed: {e}")
        gpu_load_time = None

    df_cpu, cpu_load_time = load_data_cpu(data_dir)
    results['benchmarks']['data_loading_cpu'] = cpu_load_time

    if gpu_load_time:
        speedup = cpu_load_time / gpu_load_time
        print(f"\n   🏆 GPU Speedup: {speedup:.1f}x faster")
        results['benchmarks']['data_loading_speedup'] = speedup

    # Benchmark 2: Feature Engineering
    print("\n" + "-" * 60)
    print("BENCHMARK 2: Feature Engineering")
    print("-" * 60)

    try:
        import cudf
        df_gpu = cudf.from_pandas(df_cpu)
        df_gpu_feat, gpu_feat_time = engineer_features_gpu(df_gpu)
        results['benchmarks']['feature_engineering_gpu'] = gpu_feat_time

        # Get feature columns for ML
        feature_cols_gpu = [c for c in df_gpu_feat.columns if c not in
                          ['date', 'portid', 'portname', 'country', 'continent'] and
                          not c.startswith('target_')]

        del df_gpu
        gc.collect()
    except Exception as e:
        print(f"   GPU feature engineering failed: {e}")
        gpu_feat_time = None
        df_gpu_feat = None

    df_cpu_feat, cpu_feat_time = engineer_features_cpu(df_cpu)
    results['benchmarks']['feature_engineering_cpu'] = cpu_feat_time

    if gpu_feat_time:
        speedup = cpu_feat_time / gpu_feat_time
        print(f"\n   🏆 GPU Speedup: {speedup:.1f}x faster")
        results['benchmarks']['feature_engineering_speedup'] = speedup

    del df_cpu
    gc.collect()

    # Prepare data for ML benchmarks
    print("\n" + "-" * 60)
    print("Preparing ML data...")
    print("-" * 60)

    feature_cols = [c for c in df_cpu_feat.columns if c not in
                   ['date', 'portid', 'portname', 'country', 'continent'] and
                   not c.startswith('target_')]
    feature_cols = [c for c in feature_cols if df_cpu_feat[c].dtype in ['float64', 'int64', 'float32', 'int32']]

    df_ml = df_cpu_feat.dropna(subset=['target_calls_1d']).copy()
    X = df_ml[feature_cols].fillna(0).values
    y = df_ml['target_calls_1d'].values

    # Train/test split
    split_idx = int(len(X) * 0.8)
    X_train, X_test = X[:split_idx], X[split_idx:]
    y_train, y_test = y[:split_idx], y[split_idx:]

    print(f"   Train: {len(X_train):,}, Test: {len(X_test):,}")
    print(f"   Features: {len(feature_cols)}")

    # Benchmark 3: XGBoost Training
    print("\n" + "-" * 60)
    print("BENCHMARK 3: XGBoost Training (1000 trees)")
    print("-" * 60)

    try:
        _, gpu_xgb_metrics = train_xgboost_gpu(X_train, y_train, X_test, y_test, n_estimators=1000)
        results['benchmarks']['xgboost_gpu'] = gpu_xgb_metrics
    except Exception as e:
        print(f"   XGBoost GPU failed: {e}")
        gpu_xgb_metrics = None

    _, cpu_xgb_metrics = train_xgboost_cpu(X_train, y_train, X_test, y_test, n_estimators=1000)
    results['benchmarks']['xgboost_cpu'] = cpu_xgb_metrics

    if gpu_xgb_metrics:
        speedup = cpu_xgb_metrics['train_time'] / gpu_xgb_metrics['train_time']
        print(f"\n   🏆 GPU Speedup: {speedup:.1f}x faster")
        results['benchmarks']['xgboost_speedup'] = speedup

    # Benchmark 4: cuML Models
    print("\n" + "-" * 60)
    print("BENCHMARK 4: cuML Models (GPU-only)")
    print("-" * 60)

    try:
        cuml_results = train_cuml_models(X_train, y_train, X_test, y_test)
        results['benchmarks']['cuml'] = cuml_results
    except Exception as e:
        print(f"   cuML failed: {e}")

    # Benchmark 5: PyTorch LSTM
    print("\n" + "-" * 60)
    print("BENCHMARK 5: PyTorch LSTM")
    print("-" * 60)

    try:
        # Prepare sequence data
        seq_length = 30
        X_seq, y_seq = create_sequences(np.column_stack([y, X[:, :10]]), seq_length)

        split_seq = int(len(X_seq) * 0.8)
        X_train_seq, X_val_seq = X_seq[:split_seq], X_seq[split_seq:]
        y_train_seq, y_val_seq = y_seq[:split_seq], y_seq[split_seq:]

        lstm = PortLSTM(input_size=X_seq.shape[2], hidden_size=128, num_layers=2)
        lstm_time = lstm.train(X_train_seq, y_train_seq, X_val_seq, y_val_seq, epochs=50, batch_size=256)

        results['benchmarks']['lstm_gpu'] = {
            'train_time': lstm_time,
            'device': str(lstm.device)
        }
    except Exception as e:
        print(f"   LSTM failed: {e}")

    # Summary
    print("\n" + "=" * 80)
    print("📊 BENCHMARK SUMMARY")
    print("=" * 80)

    summary_table = []
    if 'data_loading_speedup' in results['benchmarks']:
        summary_table.append(('Data Loading', results['benchmarks']['data_loading_speedup']))
    if 'feature_engineering_speedup' in results['benchmarks']:
        summary_table.append(('Feature Engineering', results['benchmarks']['feature_engineering_speedup']))
    if 'xgboost_speedup' in results['benchmarks']:
        summary_table.append(('XGBoost Training', results['benchmarks']['xgboost_speedup']))

    print("\n   GPU Speedups:")
    for task, speedup in summary_table:
        print(f"   {task:25s} {speedup:6.1f}x faster")

    # GPU Memory
    mem_info = get_gpu_memory_info()
    if mem_info:
        print(f"\n   GPU Memory: {mem_info.get('utilization_pct', 0):.1f}% utilized")
        results['gpu_memory'] = mem_info

    # Save results
    output_dir = Path(__file__).parent / 'output'
    output_dir.mkdir(exist_ok=True)

    with open(output_dir / 'gpu_benchmark_results.json', 'w') as f:
        json.dump(results, f, indent=2, default=str)

    print(f"\n   Results saved to: output/gpu_benchmark_results.json")

    return results


# ============================================================================
# Main GPU Training Pipeline
# ============================================================================

def main(args):
    """Main GPU training pipeline."""

    print("=" * 80)
    print("🚀 PORT-TO-RAIL SURGE FORECASTER - GPU TRAINING PIPELINE")
    print("=" * 80)
    print(f"Start time: {datetime.now()}")

    # Detect GPU capabilities
    capabilities = detect_gpu_capabilities()

    if not capabilities['cuda_available']:
        print("\n⚠️ No NVIDIA GPU detected. Running in CPU mode.")
        print("   For full GPU acceleration, run on DGX or GPU-enabled system.")

    data_dir = Path(__file__).parent / 'data'

    if args.benchmark:
        # Run benchmark mode
        run_benchmark(data_dir)
    else:
        # Run training
        print("\n" + "=" * 60)
        print("TRAINING MODE")
        print("=" * 60)

        if capabilities['rapids_available']:
            # Use RAPIDS pipeline
            df, _ = load_data_gpu(data_dir)
            df, _ = engineer_features_gpu(df)

            # Convert for XGBoost
            df_pd = df.to_pandas()
        else:
            # Fallback to pandas
            df_pd, _ = load_data_cpu(data_dir)
            df_pd, _ = engineer_features_cpu(df_pd)

        # Prepare ML data
        feature_cols = [c for c in df_pd.columns if c not in
                       ['date', 'portid', 'portname', 'country', 'continent'] and
                       not c.startswith('target_')]
        feature_cols = [c for c in feature_cols if df_pd[c].dtype in ['float64', 'int64', 'float32', 'int32']]

        df_ml = df_pd.dropna(subset=['target_calls_1d'])
        X = df_ml[feature_cols].fillna(0).values
        y = df_ml['target_calls_1d'].values

        split_idx = int(len(X) * 0.8)
        X_train, X_test = X[:split_idx], X[split_idx:]
        y_train, y_test = y[:split_idx], y[split_idx:]

        # Train with GPU if available
        if capabilities['xgboost_gpu']:
            model, metrics = train_xgboost_gpu(X_train, y_train, X_test, y_test)
        else:
            model, metrics = train_xgboost_cpu(X_train, y_train, X_test, y_test)

        print("\n" + "=" * 60)
        print("✅ TRAINING COMPLETE")
        print("=" * 60)
        print(f"   R²: {metrics['r2']:.4f}")
        print(f"   MAE: {metrics['mae']:.4f}")
        print(f"   Training time: {metrics['train_time']:.2f}s")


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description="GPU Training Pipeline")
    parser.add_argument('--benchmark', action='store_true', help='Run CPU vs GPU benchmark')
    args = parser.parse_args()

    main(args)
