# 🚢 Port-to-Rail Surge Forecaster - Complete Solution

## DGX Spark Frontier Hackathon 2025 | Glīd Partner Challenge

A GPU-accelerated system for predicting port surges and optimizing rail dispatch across US multimodal freight networks.

---

## 📋 Challenge Requirements Met

| Requirement | Status | Implementation |
|-------------|--------|----------------|
| 24-72h port surge predictions | ✅ | XGBoost multi-horizon model |
| Early warnings for rail congestion | ✅ | Z-score anomaly detection |
| First-mile drayage forecasting | ✅ | Distance-based time/cost estimation |
| Ideal rail dispatch windows | ✅ | Risk-based window optimization |
| Terminal repositioning routes | ✅ | Demand balancing algorithm |
| Real-time visualization dashboard | ✅ | Industrial command center UI |

---

## 🏗️ Architecture

```
┌─────────────────────────────────────────────────────────────────┐
│                    FRONTEND (Industrial HUD)                     │
│  • Real-time port map with risk indicators                      │
│  • Surge alerts panel                                           │
│  • Dispatch window recommendations                              │
│  • Terminal utilization monitoring                              │
│  • Truck repositioning suggestions                              │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    FASTAPI BACKEND                               │
│  /dashboard  - Complete dashboard data                          │
│  /predict    - Surge predictions                                │
│  /optimize   - Dispatch optimization                            │
│  /alerts     - Active surge alerts                              │
│  /ports      - Port information                                 │
│  /stats      - Aggregate statistics                             │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    MODEL LAYER                                   │
│  • ChampionModel (XGBoost ensemble)                             │
│  • Feature Engineering (64+ features)                           │
│  • Optimization Engine (dispatch windows)                       │
└─────────────────────────────────────────────────────────────────┘
                              │
                              ▼
┌─────────────────────────────────────────────────────────────────┐
│                    DATA LAYER                                    │
│  • IMF PortWatch (5M+ records)                                  │
│  • NTAD Rail Network (250K nodes)                               │
│  • Spatial joins (port → terminal)                              │
└─────────────────────────────────────────────────────────────────┘
```

---

## 🚀 Quick Start

### 1. Install Dependencies

```bash
cd logistics
pip install -r requirements.txt
```

### 2. Train the Champion Model

```bash
# CPU mode
python train_champion_model.py

# GPU mode (DGX/CUDA)
python train_champion_model.py --gpu
```

### 3. Start the API Server

```bash
uvicorn api.main:app --reload --port 8000
```

### 4. Open the Dashboard

```bash
open frontend/index.html
# Or navigate to: http://localhost:8000/docs for API docs
```

---

## 📁 Project Structure

```
logistics/
├── src/                          # Model package
│   ├── __init__.py
│   ├── config.py                 # Configuration settings
│   ├── data_loader.py            # Data loading utilities
│   ├── feature_engineering.py    # 64+ feature engineering
│   ├── model.py                  # ChampionModel (XGBoost)
│   ├── optimization.py           # Dispatch optimization
│   └── inference.py              # Inference pipeline
│
├── api/                          # FastAPI backend
│   ├── __init__.py
│   └── main.py                   # REST API endpoints
│
├── frontend/                     # Dashboard UI
│   └── index.html                # Industrial command center
│
├── models/                       # Saved models (after training)
│   ├── champion_calls_1d.json
│   ├── champion_calls_3d.json
│   ├── champion_calls_7d.json
│   └── champion_metadata.json
│
├── output/                       # Processed data
│   ├── champion_features.parquet
│   ├── port_terminal_mapping.csv
│   ├── surge_analysis.csv
│   └── training_report.json
│
├── data/                         # Raw data (not in repo)
│   ├── Daily_Port_Activity_Data_and_Trade_Estimates.csv
│   ├── PortWatch_ports_database.csv
│   └── NTAD_Rail_Network_Nodes.geojson
│
├── train_champion_model.py       # Training script
├── requirements.txt              # Python dependencies
└── README_SOLUTION.md            # This file
```

---

## 🔧 Model Features (64+ Features)

### Temporal Features
- `day_of_week`, `month`, `quarter`, `day_of_year`
- `is_weekend`, `is_month_end`, `is_quarter_end`
- Cyclical encoding: `dow_sin`, `dow_cos`, `month_sin`, `month_cos`

### Rolling Statistics
- Moving averages: `ma7`, `ma14`, `ma30`
- Standard deviations: `std7`, `std14`, `std30`
- Z-scores: `zscore_7d`, `zscore_30d`

### Lag Features
- Port calls: `calls_lag1`, `calls_lag3`, `calls_lag7`, `calls_lag14`
- Imports: `import_lag1`, `import_lag3`, `import_lag7`, `import_lag14`

### Surge Indicators
- `surge_2std`, `surge_3std` (statistical anomalies)
- `surge_relative` (percentage above moving average)
- `momentum_3d`, `momentum_7d`, `momentum_14d`

### Volume & Composition
- `total_import`, `total_export`, `total_volume`
- `container_pct`, `tanker_pct`, `bulk_pct`
- `import_export_ratio`, `volume_per_call`

### Port Characteristics
- `activity_rank`, `port_tier`

---

## 📊 API Endpoints

| Endpoint | Method | Description |
|----------|--------|-------------|
| `/` | GET | API info |
| `/health` | GET | System health check |
| `/dashboard` | GET | Complete dashboard data |
| `/ports` | GET | List all ports |
| `/ports/{name}` | GET | Specific port details |
| `/predictions` | GET | Surge predictions |
| `/optimize` | GET | Dispatch optimization |
| `/alerts` | GET | Active surge alerts |
| `/feature-importance` | GET | Model feature importance |
| `/stats` | GET | Aggregate statistics |

### Example API Call

```bash
curl http://localhost:8000/predictions?port=Houston&days_ahead=1
```

---

## 🖥️ GPU Configuration (DGX/128GB)

For maximum performance on DGX with 128GB GPU memory:

```python
# In src/config.py
MODEL_CONFIG = ModelConfig(
    max_depth=12,
    n_estimators=1000,
    learning_rate=0.05,
    tree_method='gpu_hist',
    device='cuda',
    max_bin=1024,      # Higher for GPU
    grow_policy='lossguide',
)
```

Training with GPU:
```bash
python train_champion_model.py --gpu
```

---

## 🎨 Frontend Features

The Industrial Command Center dashboard includes:

1. **Real-time Port Map**
   - Color-coded risk indicators (green/amber/red)
   - Pulsing animations for high-risk ports
   - Interactive tooltips

2. **Surge Alerts Panel**
   - Sorted by probability
   - Severity indicators
   - Expected call volumes

3. **Dispatch Windows**
   - Optimal time recommendations
   - Risk scores
   - Expected truck counts

4. **Terminal Utilization**
   - Capacity percentages
   - Status indicators (CRITICAL/HIGH/NORMAL/LOW)
   - Visual progress bars

5. **Truck Repositioning**
   - From/to terminal pairs
   - Distance and urgency
   - Reason explanations

6. **Model Insights**
   - Feature importance chart
   - Top predictive factors

---

## 📈 Performance Metrics

Expected model performance (after proper training):

| Target | MAE | R² | Notes |
|--------|-----|-----|-------|
| calls_1d | ~1.2 | ~0.70 | 24h forecast |
| calls_3d | ~1.5 | ~0.65 | 72h forecast |
| calls_7d | ~2.0 | ~0.55 | 7-day forecast |
| surge_1d | - | AUC ~0.85 | Binary classification |

---

## 🔮 Future Enhancements

1. **Real-time Data Integration**
   - WebSocket connections for live updates
   - AIS vessel tracking integration

2. **Advanced Optimization**
   - Multi-objective optimization (cost vs time vs utilization)
   - Route optimization with traffic data

3. **Enhanced Predictions**
   - Deep learning models (LSTM/Transformer)
   - Weather API integration for weather-adjusted forecasts

4. **Expanded Coverage**
   - International ports
   - Cross-border rail connections

---

## 👥 Team

**Ultra Ego DGX** - DGX Spark Frontier Hackathon 2025

---

## 📄 License

MIT
