# Model Improvement Plan: Port-to-Rail Surge Forecaster

## Current State Analysis

### Performance Metrics (Baseline)
| Target | Metric | Value | Assessment |
|--------|--------|-------|------------|
| calls_1d | R² | 0.394 | Moderate - needs improvement |
| calls_1d | MAE | 1.22 | Acceptable for port call counts |
| calls_1d | MAPE | 64.6% | High - struggling with low-volume ports |
| calls_3d | R² | 0.343 | Declining accuracy |
| calls_7d | R² | 0.318 | Further decline at longer horizons |
| surge_1d | AUC | 0.881 | Good discrimination |
| surge_1d | Recall | 3% | **Critical issue** - missing 97% of surges |

### Current Feature Set (81 features)
- **Temporal**: day_of_week, month, quarter, cyclical encodings
- **Volume**: total_import, total_export, volume_per_call
- **Rolling Statistics**: ma7, ma14, ma30, std7, std14, std30
- **Lag Features**: 1, 3, 7, 14 day lags
- **Surge Indicators**: zscore_7d, zscore_30d, surge_2std

### Key Issues Identified
1. **Severe class imbalance**: Only ~1% of days are surge days
2. **No external signals**: Model only sees historical port data
3. **Missing upstream indicators**: Chokepoint delays not incorporated
4. **No disruption awareness**: Weather/events not factored in

---

## Improvement Strategies

### Phase 1: Integrate Unused Data Sources (High Impact)

#### 1.1 Chokepoint Data Integration
**File**: `data/Daily_Chokepoints_Data.csv` (12.5 MB)

**Value**: Chokepoint congestion (Suez, Panama) causes downstream port surges 5-15 days later.

**Features to Add**:
```python
# Chokepoint flow features (lead indicators)
chokepoint_features = [
    'suez_container_count',      # Daily container vessels through Suez
    'suez_capacity_total',       # Total cargo capacity through Suez
    'panama_container_count',    # Same for Panama
    'panama_capacity_total',
    'suez_ma7',                  # 7-day moving average
    'suez_pct_change_7d',        # Week-over-week change
    'suez_anomaly_flag',         # Z-score > 2 at chokepoint
]
```

**Implementation**:
```python
def add_chokepoint_features(df, chokepoint_df):
    """Add chokepoint traffic as leading indicators."""
    # Aggregate by date
    choke_daily = chokepoint_df.groupby('date').agg({
        'n_container': 'sum',
        'capacity': 'sum',
        'n_total': 'sum'
    }).reset_index()

    # Add rolling stats
    choke_daily['choke_ma7'] = choke_daily['n_container'].rolling(7).mean()
    choke_daily['choke_pct_change'] = choke_daily['n_container'].pct_change(7)

    # Lag chokepoint data (vessels take time to reach ports)
    for lag in [5, 7, 10, 14]:
        choke_daily[f'choke_container_lag{lag}'] = choke_daily['n_container'].shift(lag)

    # Merge with port data
    return df.merge(choke_daily, on='date', how='left')
```

#### 1.2 Disruption Events Integration
**File**: `data/portwatch_disruptions_database.csv`

**Value**: Tropical cyclones, floods, and other events directly impact port operations.

**Features to Add**:
```python
disruption_features = [
    'active_disruptions',       # Count of active events
    'tc_active',                # Tropical cyclone flag
    'flood_active',             # Flood event flag
    'disruption_severity',      # Encoded severity (RED=3, ORANGE=2, etc.)
    'days_since_disruption',    # Recovery indicator
    'nearby_port_affected',     # If connected ports disrupted
]
```

**Implementation**:
```python
def add_disruption_features(df, disruptions_df):
    """Add disruption event features."""
    # Parse dates
    disruptions_df['fromdate'] = pd.to_datetime(disruptions_df['fromdate'])
    disruptions_df['todate'] = pd.to_datetime(disruptions_df['todate'])

    # Create daily disruption flags
    date_range = pd.date_range(df['date'].min(), df['date'].max())

    disruption_daily = []
    for date in date_range:
        active = disruptions_df[
            (disruptions_df['fromdate'] <= date) &
            (disruptions_df['todate'] >= date)
        ]
        disruption_daily.append({
            'date': date,
            'active_disruptions': len(active),
            'tc_active': (active['eventtype'] == 'TC').sum(),
            'flood_active': (active['eventtype'] == 'FL').sum(),
            'max_severity': 3 if 'RED' in active['alertlevel'].values else
                           2 if 'ORANGE' in active['alertlevel'].values else 1
        })

    return df.merge(pd.DataFrame(disruption_daily), on='date', how='left')
```

#### 1.3 Freight Cost/Volume Data
**File**: `data/fFreight.csv` (7.7 MB)

**Value**: Freight volume and revenue indicate economic activity that drives port traffic.

**Features to Add**:
```python
freight_features = [
    'freight_volume_7d',        # Rolling 7-day freight weight
    'freight_revenue_7d',       # Rolling revenue
    'freight_goods_value_7d',   # Value of goods
    'freight_volume_trend',     # Is freight increasing?
]
```

---

### Phase 2: Address Class Imbalance (Critical for Surge Detection)

Current surge recall is **3%** - the model predicts "no surge" almost always.

#### 2.1 SMOTE Oversampling
```python
from imblearn.over_sampling import SMOTE

def train_surge_with_smote(X, y_surge):
    """Train surge classifier with oversampling."""
    smote = SMOTE(sampling_strategy=0.3, random_state=42)  # 30% minority
    X_resampled, y_resampled = smote.fit_resample(X, y_surge)

    model = XGBClassifier(
        scale_pos_weight=1,  # SMOTE handles imbalance
        max_depth=8,
        n_estimators=500,
        learning_rate=0.05
    )
    model.fit(X_resampled, y_resampled)
    return model
```

#### 2.2 Class Weights
```python
# Calculate weight based on class ratio
pos_weight = (y_surge == 0).sum() / (y_surge == 1).sum()  # ~99

model = XGBClassifier(
    scale_pos_weight=pos_weight,  # Penalize missing surges heavily
    max_depth=10,
    n_estimators=500
)
```

#### 2.3 Threshold Optimization
```python
from sklearn.metrics import precision_recall_curve

def optimize_threshold(model, X_test, y_test):
    """Find threshold that maximizes F1 for surge detection."""
    y_prob = model.predict_proba(X_test)[:, 1]

    precisions, recalls, thresholds = precision_recall_curve(y_test, y_prob)
    f1_scores = 2 * (precisions * recalls) / (precisions + recalls + 1e-8)

    best_idx = np.argmax(f1_scores)
    best_threshold = thresholds[best_idx]  # Likely ~0.1-0.2 instead of 0.5

    return best_threshold
```

#### 2.4 Focal Loss (Advanced)
```python
# Custom focal loss for XGBoost - down-weights easy negatives
def focal_loss_obj(y_true, y_pred, gamma=2.0, alpha=0.25):
    """Focal loss objective for XGBoost."""
    p = 1.0 / (1.0 + np.exp(-y_pred))
    grad = alpha * (1 - p) ** gamma * (p - y_true)
    hess = alpha * (1 - p) ** gamma * p * (1 - p) * (gamma * (p - y_true) / (1 - p + 1e-8) + 1)
    return grad, hess
```

---

### Phase 3: Feature Engineering Enhancements

#### 3.1 Cross-Port Features
```python
def add_cross_port_features(df):
    """Add features that capture regional/network effects."""

    # Regional aggregates (e.g., West Coast, Gulf, East Coast)
    region_mapping = {
        'Los Angeles': 'west', 'Long Beach': 'west', 'Oakland': 'west',
        'Houston': 'gulf', 'New Orleans': 'gulf',
        'New York': 'east', 'Savannah': 'east', 'Charleston': 'east'
    }
    df['region'] = df['portname'].map(region_mapping).fillna('other')

    # Regional average activity (other ports in same region)
    df['regional_avg_calls'] = df.groupby(['date', 'region'])['portcalls'].transform('mean')
    df['vs_regional_avg'] = df['portcalls'] / df['regional_avg_calls']

    # National aggregate
    df['national_total_calls'] = df.groupby('date')['portcalls'].transform('sum')
    df['port_share'] = df['portcalls'] / df['national_total_calls']

    return df
```

#### 3.2 Volatility Features
```python
def add_volatility_features(df):
    """Add volatility and trend features."""

    # Volatility (coefficient of variation)
    df['cv_7d'] = df['std7'] / df['ma7'].replace(0, 1)
    df['cv_30d'] = df['std30'] / df['ma30'].replace(0, 1)

    # Trend strength (linear regression slope)
    df['trend_7d'] = df.groupby('portname')['portcalls'].transform(
        lambda x: x.rolling(7).apply(lambda y: np.polyfit(range(len(y)), y, 1)[0] if len(y) == 7 else 0)
    )

    # Mean reversion indicator
    df['deviation_from_mean'] = (df['portcalls'] - df['ma30']) / df['std30'].replace(0, 1)

    return df
```

#### 3.3 Interaction Features
```python
def add_interaction_features(df):
    """Add important feature interactions."""

    # Volume × Volatility (high volume + high volatility = high risk)
    df['volume_volatility'] = df['total_volume'] * df['cv_7d']

    # Weekend × Volume (weekend surges are different)
    df['weekend_volume'] = df['is_weekend'] * df['total_volume']

    # Month-end × Import (shipping deadlines)
    df['monthend_import'] = df['is_month_end'] * df['total_import']

    return df
```

---

### Phase 4: Model Architecture Improvements

#### 4.1 Ensemble Approach
```python
from sklearn.ensemble import VotingRegressor
from lightgbm import LGBMRegressor
from catboost import CatBoostRegressor

def create_ensemble():
    """Create ensemble of diverse models."""

    xgb = XGBRegressor(max_depth=10, n_estimators=500, learning_rate=0.05)
    lgb = LGBMRegressor(max_depth=10, n_estimators=500, learning_rate=0.05)
    cat = CatBoostRegressor(depth=10, iterations=500, learning_rate=0.05, verbose=0)

    ensemble = VotingRegressor([
        ('xgb', xgb),
        ('lgb', lgb),
        ('cat', cat)
    ])

    return ensemble
```

#### 4.2 Stacking with Meta-Learner
```python
from sklearn.ensemble import StackingRegressor
from sklearn.linear_model import Ridge

def create_stacking_model():
    """Create stacking ensemble with meta-learner."""

    estimators = [
        ('xgb', XGBRegressor(max_depth=8, n_estimators=300)),
        ('lgb', LGBMRegressor(max_depth=8, n_estimators=300)),
    ]

    stacking = StackingRegressor(
        estimators=estimators,
        final_estimator=Ridge(),
        cv=5
    )

    return stacking
```

#### 4.3 Separate Models per Port Tier
```python
def train_tiered_models(df, feature_cols):
    """Train separate models for high/medium/low volume ports."""

    models = {}

    for tier in [1, 2, 3, 4]:
        tier_data = df[df['port_tier'] == tier]
        X = tier_data[feature_cols].fillna(0)
        y = tier_data['target_calls_1d']

        model = XGBRegressor(
            max_depth=8 if tier <= 2 else 6,  # Deeper for major ports
            n_estimators=500 if tier <= 2 else 300
        )
        model.fit(X, y)
        models[tier] = model

    return models
```

---

### Phase 5: Hyperparameter Optimization

#### 5.1 Optuna Bayesian Optimization
```python
import optuna

def optimize_xgboost(X_train, y_train, X_val, y_val):
    """Bayesian hyperparameter optimization."""

    def objective(trial):
        params = {
            'max_depth': trial.suggest_int('max_depth', 4, 15),
            'n_estimators': trial.suggest_int('n_estimators', 100, 1000),
            'learning_rate': trial.suggest_float('learning_rate', 0.01, 0.3, log=True),
            'min_child_weight': trial.suggest_int('min_child_weight', 1, 10),
            'subsample': trial.suggest_float('subsample', 0.6, 1.0),
            'colsample_bytree': trial.suggest_float('colsample_bytree', 0.6, 1.0),
            'gamma': trial.suggest_float('gamma', 0, 5),
            'reg_alpha': trial.suggest_float('reg_alpha', 1e-8, 10, log=True),
            'reg_lambda': trial.suggest_float('reg_lambda', 1e-8, 10, log=True),
        }

        model = XGBRegressor(**params, tree_method='hist')
        model.fit(X_train, y_train)
        y_pred = model.predict(X_val)

        return mean_squared_error(y_val, y_pred)

    study = optuna.create_study(direction='minimize')
    study.optimize(objective, n_trials=100)

    return study.best_params
```

---

### Phase 6: Validation Strategy Improvements

#### 6.1 Time-Series Cross-Validation
```python
from sklearn.model_selection import TimeSeriesSplit

def time_series_cv(X, y, n_splits=5):
    """Proper time series cross-validation."""

    tscv = TimeSeriesSplit(n_splits=n_splits)
    scores = []

    for train_idx, test_idx in tscv.split(X):
        X_train, X_test = X.iloc[train_idx], X.iloc[test_idx]
        y_train, y_test = y.iloc[train_idx], y.iloc[test_idx]

        model = XGBRegressor()
        model.fit(X_train, y_train)

        y_pred = model.predict(X_test)
        scores.append(r2_score(y_test, y_pred))

    return np.mean(scores), np.std(scores)
```

#### 6.2 Walk-Forward Validation
```python
def walk_forward_validation(df, train_months=12, test_months=1):
    """Walk-forward validation for realistic performance estimation."""

    results = []
    dates = df['date'].unique()

    for i in range(0, len(dates) - train_months - test_months, test_months):
        train_end = dates[i + train_months]
        test_end = dates[i + train_months + test_months]

        train_data = df[df['date'] <= train_end]
        test_data = df[(df['date'] > train_end) & (df['date'] <= test_end)]

        # Train and evaluate
        model.fit(train_data[features], train_data[target])
        predictions = model.predict(test_data[features])

        results.append({
            'period': test_end,
            'r2': r2_score(test_data[target], predictions),
            'mae': mean_absolute_error(test_data[target], predictions)
        })

    return pd.DataFrame(results)
```

---

## Implementation Roadmap

### Step 1: Data Integration (Estimated Impact: +10-15% R²)
```
1. Load and parse Daily_Chokepoints_Data.csv
2. Load and parse portwatch_disruptions_database.csv
3. Create date-aligned merge with port activity data
4. Add lagged chokepoint features (5, 7, 10, 14 days)
5. Add active disruption flags
```

### Step 2: Class Imbalance (Estimated Impact: +40% Recall)
```
1. Implement SMOTE oversampling for surge target
2. Add class weights to surge classifier
3. Optimize classification threshold using PR curve
4. Target: Recall > 50% while maintaining Precision > 30%
```

### Step 3: Feature Engineering (Estimated Impact: +5-10% R²)
```
1. Add cross-port/regional features
2. Add volatility features
3. Add interaction features
4. Feature selection using importance + correlation
```

### Step 4: Model Improvements (Estimated Impact: +5% R²)
```
1. Hyperparameter tuning with Optuna
2. Test LightGBM and CatBoost
3. Create ensemble if beneficial
4. Implement tiered models if heterogeneity detected
```

---

## Expected Outcomes

| Metric | Current | Target | Improvement |
|--------|---------|--------|-------------|
| calls_1d R² | 0.39 | 0.55-0.65 | +40-65% |
| calls_1d MAE | 1.22 | 0.90-1.00 | -18-26% |
| calls_3d R² | 0.34 | 0.50-0.55 | +47-62% |
| surge_1d Recall | 3% | 50-60% | +1567-1900% |
| surge_1d Precision | 44% | 35-45% | Maintained |

---

## Quick Wins (Can Implement Today)

1. **Add class weights for surge** - Single line change, huge recall improvement
2. **Lower classification threshold** - From 0.5 to ~0.15 based on PR curve
3. **Add chokepoint lagged features** - High value, moderate effort
4. **Optuna hyperparameter tuning** - Automatic, runs overnight

---

## Files to Create/Modify

1. `src/data_loader.py` - Add loaders for chokepoints, disruptions, freight
2. `src/feature_engineering.py` - Add new feature functions
3. `src/model.py` - Add SMOTE, class weights, threshold optimization
4. `train_enhanced_model.py` - New training script with all improvements
5. `requirements.txt` - Add imbalanced-learn, optuna, lightgbm, catboost
