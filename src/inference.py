"""
Inference pipeline for Port-to-Rail Surge Forecaster.
Provides real-time prediction capabilities.
"""

from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional, Union

import numpy as np
import pandas as pd

from .config import MODEL_DIR, OUTPUT_DIR, INFERENCE_CONFIG
from .model import ChampionModel
from .feature_engineering import engineer_all_features, get_feature_columns
from .optimization import optimize_dispatch, DispatchOptimizer


class InferencePipeline:
    """
    Production inference pipeline for port surge forecasting.

    Features:
    - Load trained models
    - Real-time feature engineering
    - Multi-horizon predictions
    - Dispatch optimization
    - Caching for performance
    """

    def __init__(
        self,
        model_path: Path = MODEL_DIR,
        model_prefix: str = 'champion'
    ):
        self.model_path = Path(model_path)
        self.model_prefix = model_prefix
        self.model: Optional[ChampionModel] = None
        self.port_terminal_map: Optional[pd.DataFrame] = None

        # Cache
        self._cache: Dict = {}
        self._cache_timestamp: Optional[datetime] = None

    def load(self) -> 'InferencePipeline':
        """Load model and supporting data."""

        print("Loading inference pipeline...")

        # Load model
        self.model = ChampionModel.load(self.model_path, self.model_prefix)
        print(f"  ✅ Model loaded ({len(self.model.models)} targets)")

        # Load port-terminal mapping
        mapping_path = OUTPUT_DIR / "port_terminal_mapping.csv"
        if mapping_path.exists():
            self.port_terminal_map = pd.read_csv(mapping_path)
            print(f"  ✅ Port-terminal mapping loaded ({len(self.port_terminal_map)} ports)")

        return self

    def predict(
        self,
        data: pd.DataFrame,
        include_optimization: bool = True
    ) -> Dict:
        """
        Run inference on input data.

        Args:
            data: DataFrame with port activity data
            include_optimization: Include dispatch optimization

        Returns:
            Dictionary with predictions and recommendations
        """

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Feature engineering
        print("Engineering features...")
        features_df = engineer_all_features(data, add_targets=False)

        # Get feature columns
        feature_cols = self.model.feature_cols

        # Prepare input
        X = features_df[feature_cols].fillna(0)

        # Make predictions
        print("Making predictions...")
        predictions = self.model.predict_all(X)

        # Add metadata
        predictions['portname'] = features_df['portname'].values if 'portname' in features_df.columns else 'Unknown'
        predictions['date'] = features_df['date'].values if 'date' in features_df.columns else datetime.now()

        result = {
            'predictions': predictions.to_dict('records'),
            'metrics': self.model.metrics,
            'timestamp': datetime.now().isoformat()
        }

        # Dispatch optimization
        if include_optimization and self.port_terminal_map is not None:
            print("Generating optimization recommendations...")
            result['optimization'] = optimize_dispatch(
                self.port_terminal_map,
                predictions,
                datetime.now()
            )

        return result

    def predict_port(
        self,
        port_name: str,
        historical_data: pd.DataFrame,
        horizon_days: int = 7
    ) -> Dict:
        """
        Predict for a specific port.

        Args:
            port_name: Name of the port
            historical_data: Recent historical data for the port
            horizon_days: Days to forecast

        Returns:
            Port-specific predictions
        """

        if self.model is None:
            raise RuntimeError("Model not loaded. Call load() first.")

        # Filter to port
        port_data = historical_data[historical_data['portname'] == port_name].copy()

        if port_data.empty:
            return {'error': f'No data found for port: {port_name}'}

        # Engineer features
        features_df = engineer_all_features(port_data, add_targets=False)

        # Get latest row for prediction
        latest = features_df.iloc[-1:].copy()
        X = latest[self.model.feature_cols].fillna(0)

        # Predictions with confidence
        result = {
            'port_name': port_name,
            'forecast_date': datetime.now().isoformat(),
            'predictions': {}
        }

        for target_name in self.model.models.keys():
            pred, lower, upper = self.model.predict(X, target_name, return_confidence=True)
            result['predictions'][target_name] = {
                'value': float(pred[0]),
                'lower_bound': float(lower[0]),
                'upper_bound': float(upper[0])
            }

        # Port terminal info
        if self.port_terminal_map is not None:
            port_info = self.port_terminal_map[
                self.port_terminal_map['portname'] == port_name
            ]
            if not port_info.empty:
                result['terminal'] = port_info.iloc[0].to_dict()

        return result

    def get_surge_alerts(
        self,
        predictions: pd.DataFrame,
        threshold: float = 0.5
    ) -> List[Dict]:
        """Get ports with high surge probability."""

        alerts = []

        surge_col = 'pred_surge_1d' if 'pred_surge_1d' in predictions.columns else None

        if surge_col is None:
            return alerts

        high_risk = predictions[predictions[surge_col] > threshold]

        for _, row in high_risk.iterrows():
            alerts.append({
                'port': row.get('portname', 'Unknown'),
                'surge_probability': round(float(row[surge_col]) * 100, 1),
                'predicted_calls': round(float(row.get('pred_calls_1d', 0)), 1),
                'severity': 'HIGH' if row[surge_col] > 0.7 else 'MEDIUM',
                'timestamp': datetime.now().isoformat()
            })

        return sorted(alerts, key=lambda x: -x['surge_probability'])

    def get_system_status(self) -> Dict:
        """Get system status and health."""

        status = {
            'status': 'healthy',
            'timestamp': datetime.now().isoformat(),
            'model_loaded': self.model is not None,
            'targets_available': list(self.model.models.keys()) if self.model else [],
            'n_ports_mapped': len(self.port_terminal_map) if self.port_terminal_map is not None else 0,
        }

        if self.model:
            status['model_metrics'] = self.model.metrics

        return status


# Singleton for FastAPI
_pipeline: Optional[InferencePipeline] = None


def get_pipeline() -> InferencePipeline:
    """Get or create the inference pipeline singleton."""
    global _pipeline

    if _pipeline is None:
        _pipeline = InferencePipeline().load()

    return _pipeline


def predict_surge(data: pd.DataFrame) -> Dict:
    """Convenience function for surge prediction."""
    pipeline = get_pipeline()
    return pipeline.predict(data)
