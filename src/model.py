"""
Champion XGBoost model for Port-to-Rail Surge Forecaster.
Optimized for 128GB GPU (DGX Spark).
"""

import json
import pickle
import time
import warnings
from pathlib import Path
from typing import Dict, List, Optional, Tuple, Union

import numpy as np
import pandas as pd
from sklearn.model_selection import TimeSeriesSplit
from sklearn.metrics import mean_absolute_error, mean_squared_error, r2_score

from .config import MODEL_CONFIG, MODEL_DIR, OUTPUT_DIR

warnings.filterwarnings('ignore')

# Import XGBoost
try:
    import xgboost as xgb
    XGB_AVAILABLE = True
    print(f"✅ XGBoost {xgb.__version__}")
except ImportError:
    XGB_AVAILABLE = False
    print("⚠️ XGBoost not available")

# Try GPU
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
except ImportError:
    cudf = pd
    cp = np
    GPU_AVAILABLE = False


class ChampionModel:
    """
    Champion XGBoost ensemble for port surge forecasting.

    Features:
    - Multi-horizon forecasting (1d, 3d, 7d)
    - GPU acceleration with automatic fallback
    - Time-series cross-validation
    - Feature importance analysis
    - Confidence intervals
    """

    def __init__(self, config: Optional[MODEL_CONFIG.__class__] = None):
        self.config = config or MODEL_CONFIG
        self.models: Dict[str, xgb.Booster] = {}
        self.feature_cols: List[str] = []
        self.metrics: Dict[str, Dict] = {}
        self.feature_importance: Dict[str, pd.DataFrame] = {}

    def _get_xgb_params(self, objective: str = 'reg:squarederror') -> Dict:
        """Get XGBoost parameters optimized for GPU."""

        params = {
            'objective': objective,
            'eval_metric': self.config.eval_metric,
            'max_depth': self.config.max_depth,
            'learning_rate': self.config.learning_rate,
            'subsample': self.config.subsample,
            'colsample_bytree': self.config.colsample_bytree,
            'min_child_weight': self.config.min_child_weight,
            'gamma': self.config.gamma,
            'reg_alpha': self.config.reg_alpha,
            'reg_lambda': self.config.reg_lambda,
            'max_bin': self.config.max_bin,
            'grow_policy': self.config.grow_policy,
            'seed': self.config.seed,
        }

        # Try GPU
        if GPU_AVAILABLE:
            try:
                params['tree_method'] = 'gpu_hist'
                params['device'] = 'cuda'
            except Exception:
                params['tree_method'] = 'hist'
                params['device'] = 'cpu'
        else:
            params['tree_method'] = 'hist'
            params['device'] = 'cpu'

        return params

    def train(
        self,
        X: pd.DataFrame,
        y_dict: Dict[str, pd.Series],
        test_size: float = 0.2
    ) -> Dict[str, Dict]:
        """
        Train models for multiple forecast horizons.

        Args:
            X: Feature matrix
            y_dict: Dictionary of target series {'calls_1d': y1, 'calls_3d': y3, ...}
            test_size: Fraction for test set

        Returns:
            Dictionary of metrics per target
        """

        print("=" * 60)
        print("Training Champion Model")
        print("=" * 60)

        self.feature_cols = list(X.columns)

        # Time-based split (no shuffling for time series)
        split_idx = int(len(X) * (1 - test_size))

        results = {}

        for target_name, y in y_dict.items():
            print(f"\n📊 Training: {target_name}")
            print("-" * 40)

            # Align X and y
            valid_idx = y.notna()
            X_valid = X[valid_idx].reset_index(drop=True)
            y_valid = y[valid_idx].reset_index(drop=True)

            # Split
            split_idx_adj = int(len(X_valid) * (1 - test_size))
            X_train = X_valid.iloc[:split_idx_adj]
            X_test = X_valid.iloc[split_idx_adj:]
            y_train = y_valid.iloc[:split_idx_adj]
            y_test = y_valid.iloc[split_idx_adj:]

            print(f"  Train: {len(X_train):,}, Test: {len(X_test):,}")

            # Create DMatrix
            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            # Determine objective
            is_classification = target_name.startswith('surge')
            if is_classification:
                params = self._get_xgb_params('binary:logistic')
                params['eval_metric'] = 'auc'
            else:
                params = self._get_xgb_params('reg:squarederror')

            # Train with early stopping
            t0 = time.time()
            evals = [(dtrain, 'train'), (dtest, 'test')]

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=self.config.n_estimators,
                evals=evals,
                early_stopping_rounds=self.config.early_stopping_rounds,
                verbose_eval=50
            )

            train_time = time.time() - t0
            print(f"  ⏱️ Training time: {train_time:.1f}s")

            # Store model
            self.models[target_name] = model

            # Evaluate
            y_pred = model.predict(dtest)

            if is_classification:
                y_pred_binary = (y_pred > 0.5).astype(int)
                from sklearn.metrics import roc_auc_score, precision_score, recall_score
                metrics = {
                    'auc': roc_auc_score(y_test, y_pred),
                    'precision': precision_score(y_test, y_pred_binary, zero_division=0),
                    'recall': recall_score(y_test, y_pred_binary, zero_division=0),
                }
            else:
                metrics = {
                    'mae': mean_absolute_error(y_test, y_pred),
                    'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                    'r2': r2_score(y_test, y_pred),
                    'mape': np.mean(np.abs((y_test - y_pred) / y_test.replace(0, 1))) * 100
                }

            metrics['train_time'] = train_time
            metrics['n_samples_train'] = len(X_train)
            metrics['n_samples_test'] = len(X_test)

            self.metrics[target_name] = metrics
            results[target_name] = metrics

            # Feature importance
            importance = model.get_score(importance_type='gain')
            feat_imp = pd.DataFrame({
                'feature': list(importance.keys()),
                'importance': list(importance.values())
            }).sort_values('importance', ascending=False)
            self.feature_importance[target_name] = feat_imp

            # Print metrics
            print(f"\n  📈 Metrics:")
            for k, v in metrics.items():
                if isinstance(v, float):
                    print(f"     {k}: {v:.4f}")
                else:
                    print(f"     {k}: {v}")

        return results

    def train_with_cv(
        self,
        X: pd.DataFrame,
        y: pd.Series,
        target_name: str = 'calls_1d',
        n_splits: int = 5
    ) -> Dict:
        """Train with time-series cross-validation."""

        print(f"\n📊 Training {target_name} with {n_splits}-fold Time Series CV")

        self.feature_cols = list(X.columns)

        # Align X and y
        valid_idx = y.notna()
        X_valid = X[valid_idx].reset_index(drop=True)
        y_valid = y[valid_idx].reset_index(drop=True)

        tscv = TimeSeriesSplit(n_splits=n_splits)
        cv_metrics = []

        params = self._get_xgb_params()

        for fold, (train_idx, test_idx) in enumerate(tscv.split(X_valid)):
            print(f"\n  Fold {fold + 1}/{n_splits}")

            X_train = X_valid.iloc[train_idx]
            X_test = X_valid.iloc[test_idx]
            y_train = y_valid.iloc[train_idx]
            y_test = y_valid.iloc[test_idx]

            dtrain = xgb.DMatrix(X_train, label=y_train)
            dtest = xgb.DMatrix(X_test, label=y_test)

            model = xgb.train(
                params,
                dtrain,
                num_boost_round=200,
                evals=[(dtest, 'test')],
                early_stopping_rounds=20,
                verbose_eval=False
            )

            y_pred = model.predict(dtest)

            fold_metrics = {
                'fold': fold + 1,
                'mae': mean_absolute_error(y_test, y_pred),
                'rmse': np.sqrt(mean_squared_error(y_test, y_pred)),
                'r2': r2_score(y_test, y_pred)
            }
            cv_metrics.append(fold_metrics)

            print(f"    MAE: {fold_metrics['mae']:.3f}, R²: {fold_metrics['r2']:.3f}")

        # Store final model (trained on all data)
        dtrain_full = xgb.DMatrix(X_valid, label=y_valid)
        self.models[target_name] = xgb.train(
            params,
            dtrain_full,
            num_boost_round=self.config.n_estimators
        )

        # Aggregate CV metrics
        cv_df = pd.DataFrame(cv_metrics)
        agg_metrics = {
            'mae_mean': cv_df['mae'].mean(),
            'mae_std': cv_df['mae'].std(),
            'rmse_mean': cv_df['rmse'].mean(),
            'rmse_std': cv_df['rmse'].std(),
            'r2_mean': cv_df['r2'].mean(),
            'r2_std': cv_df['r2'].std(),
        }

        self.metrics[target_name] = agg_metrics

        print(f"\n  📈 CV Results:")
        print(f"     MAE: {agg_metrics['mae_mean']:.3f} ± {agg_metrics['mae_std']:.3f}")
        print(f"     R²:  {agg_metrics['r2_mean']:.3f} ± {agg_metrics['r2_std']:.3f}")

        return agg_metrics

    def predict(
        self,
        X: pd.DataFrame,
        target_name: str = 'calls_1d',
        return_confidence: bool = False
    ) -> Union[np.ndarray, Tuple[np.ndarray, np.ndarray, np.ndarray]]:
        """
        Make predictions.

        Args:
            X: Feature matrix
            target_name: Which model to use
            return_confidence: Return confidence intervals

        Returns:
            Predictions (and optionally lower/upper bounds)
        """

        if target_name not in self.models:
            raise ValueError(f"Model '{target_name}' not trained. Available: {list(self.models.keys())}")

        model = self.models[target_name]

        # Ensure feature alignment
        X_aligned = X[self.feature_cols].fillna(0)
        dmatrix = xgb.DMatrix(X_aligned)

        preds = model.predict(dmatrix)

        if return_confidence:
            # Estimate confidence using training metrics
            if target_name in self.metrics:
                std = self.metrics[target_name].get('rmse', self.metrics[target_name].get('rmse_mean', 1))
            else:
                std = 1

            lower = preds - 1.96 * std
            upper = preds + 1.96 * std
            return preds, lower, upper

        return preds

    def predict_all(self, X: pd.DataFrame) -> pd.DataFrame:
        """Predict all horizons."""

        results = pd.DataFrame()

        for target_name in self.models.keys():
            preds = self.predict(X, target_name)
            results[f'pred_{target_name}'] = preds

        return results

    def save(self, model_dir: Path = MODEL_DIR, prefix: str = 'champion'):
        """Save all models and metadata."""

        model_dir = Path(model_dir)
        model_dir.mkdir(exist_ok=True)

        # Save each XGBoost model
        for name, model in self.models.items():
            model_path = model_dir / f"{prefix}_{name}.json"
            model.save_model(str(model_path))
            print(f"✅ Saved: {model_path}")

        # Save metadata
        metadata = {
            'feature_cols': self.feature_cols,
            'metrics': self.metrics,
            'config': {
                'max_depth': self.config.max_depth,
                'n_estimators': self.config.n_estimators,
                'learning_rate': self.config.learning_rate,
            }
        }

        meta_path = model_dir / f"{prefix}_metadata.json"
        with open(meta_path, 'w') as f:
            json.dump(metadata, f, indent=2, default=str)
        print(f"✅ Saved: {meta_path}")

        # Save feature importance
        for name, feat_imp in self.feature_importance.items():
            imp_path = model_dir / f"{prefix}_{name}_importance.csv"
            feat_imp.to_csv(imp_path, index=False)

    @classmethod
    def load(cls, model_dir: Path = MODEL_DIR, prefix: str = 'champion') -> 'ChampionModel':
        """Load saved models."""

        model_dir = Path(model_dir)

        # Load metadata
        meta_path = model_dir / f"{prefix}_metadata.json"
        with open(meta_path, 'r') as f:
            metadata = json.load(f)

        # Create instance
        instance = cls()
        instance.feature_cols = metadata['feature_cols']
        instance.metrics = metadata['metrics']

        # Load models
        for model_file in model_dir.glob(f"{prefix}_*.json"):
            if 'metadata' in model_file.name:
                continue

            target_name = model_file.stem.replace(f"{prefix}_", "")
            model = xgb.Booster()
            model.load_model(str(model_file))
            instance.models[target_name] = model
            print(f"✅ Loaded: {model_file.name}")

        return instance

    def get_feature_importance(self, target_name: str = 'calls_1d', top_n: int = 20) -> pd.DataFrame:
        """Get top feature importance."""

        if target_name in self.feature_importance:
            return self.feature_importance[target_name].head(top_n)

        if target_name in self.models:
            importance = self.models[target_name].get_score(importance_type='gain')
            return pd.DataFrame({
                'feature': list(importance.keys()),
                'importance': list(importance.values())
            }).sort_values('importance', ascending=False).head(top_n)

        return pd.DataFrame()


def train_champion_model(
    X: pd.DataFrame,
    y_dict: Dict[str, pd.Series],
    save_path: Path = MODEL_DIR
) -> ChampionModel:
    """
    Train and save the champion model.

    Args:
        X: Feature matrix
        y_dict: Dictionary of target series
        save_path: Where to save the model

    Returns:
        Trained ChampionModel
    """

    model = ChampionModel()
    model.train(X, y_dict)
    model.save(save_path)

    return model
