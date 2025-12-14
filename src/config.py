"""Configuration settings for the Port-to-Rail Surge Forecaster."""

from pathlib import Path
from dataclasses import dataclass
from typing import Optional
import os

# Paths
PROJECT_ROOT = Path(__file__).parent.parent
DATA_DIR = PROJECT_ROOT / "data"
OUTPUT_DIR = PROJECT_ROOT / "output"
MODEL_DIR = PROJECT_ROOT / "models"

# Ensure directories exist
OUTPUT_DIR.mkdir(exist_ok=True)
MODEL_DIR.mkdir(exist_ok=True)


@dataclass
class ModelConfig:
    """XGBoost model configuration optimized for 128GB GPU."""

    # Model architecture
    max_depth: int = 12
    n_estimators: int = 500
    learning_rate: float = 0.05
    subsample: float = 0.8
    colsample_bytree: float = 0.8
    min_child_weight: int = 5
    gamma: float = 0.1
    reg_alpha: float = 0.1
    reg_lambda: float = 1.0

    # GPU settings (optimized for A100 80GB / DGX 128GB)
    tree_method: str = "gpu_hist"
    device: str = "cuda"
    max_bin: int = 512  # Higher for GPU
    grow_policy: str = "lossguide"

    # Training
    early_stopping_rounds: int = 50
    eval_metric: str = "rmse"
    seed: int = 42

    # Multi-output
    forecast_horizons: tuple = (1, 3, 7)  # 24h, 72h, 7d


@dataclass
class DataConfig:
    """Data processing configuration."""

    # Spatial
    max_drayage_distance_km: float = 100.0
    avg_truck_speed_kmh: float = 50.0
    loading_time_min: float = 30.0
    unloading_time_min: float = 30.0
    cost_per_km_usd: float = 2.50
    fixed_drayage_cost_usd: float = 150.0

    # Surge detection
    surge_zscore_threshold: float = 2.0
    surge_relative_threshold: float = 1.5  # 50% above MA

    # Features
    rolling_windows: tuple = (7, 14, 30)
    lag_periods: tuple = (1, 3, 7, 14)

    # Min activity threshold (calls/day)
    min_activity_threshold: float = 0.5


@dataclass
class InferenceConfig:
    """Inference and deployment settings."""

    # API settings
    api_host: str = "0.0.0.0"
    api_port: int = 8000

    # Batch processing
    batch_size: int = 10000

    # Caching
    cache_ttl_seconds: int = 300  # 5 minutes


# Default configs
MODEL_CONFIG = ModelConfig()
DATA_CONFIG = DataConfig()
INFERENCE_CONFIG = InferenceConfig()

# Feature columns for the model
FEATURE_COLUMNS = [
    # Port activity
    "portcalls", "portcalls_container", "portcalls_tanker", "portcalls_dry_bulk",
    "portcalls_cargo", "portcalls_roro", "portcalls_general_cargo",

    # Trade volumes
    "import_container", "export_container", "import_tanker", "export_tanker",
    "import_dry_bulk", "export_dry_bulk", "import_cargo", "export_cargo",
    "import_general_cargo", "export_general_cargo", "import_roro", "export_roro",
    "total_import", "total_export", "total_volume",

    # Moving averages
    "ma7", "ma14", "ma30",
    "import_ma7", "import_ma14", "import_ma30",
    "std7", "std14", "std30",

    # Surge indicators
    "zscore_7d", "zscore_30d",
    "momentum_3d", "momentum_7d", "momentum_14d",
    "pct_change", "pct_change_7d",

    # Lag features
    "calls_lag1", "calls_lag3", "calls_lag7", "calls_lag14",
    "import_lag1", "import_lag3", "import_lag7", "import_lag14",

    # Compositional
    "container_pct", "tanker_pct", "bulk_pct",
    "import_export_ratio", "volume_per_call",

    # Temporal
    "day_of_week", "month", "quarter", "day_of_year", "week_of_year",
    "is_weekend", "is_month_end", "is_quarter_end",
    "dow_sin", "dow_cos", "month_sin", "month_cos",

    # Port characteristics
    "activity_rank", "port_tier",
]

TARGET_COLUMNS = {
    "calls_1d": "target_calls_1d",
    "calls_3d": "target_calls_3d",
    "calls_7d": "target_calls_7d",
    "import_1d": "target_import_1d",
    "surge_1d": "target_surge_1d",
}
