# 🚢 Port-to-Rail Surge Forecaster

## DGX Spark Frontier Hackathon - December 2025

GPU-accelerated pipeline for predicting port surges and optimizing freight logistics from US ports to rail terminals.

## 🎯 Challenge

Predict port activity surges 24-72 hours in advance to optimize:
- Rail terminal capacity planning
- Drayage truck dispatch
- Container routing decisions

## 📊 Pipeline

```
Port Activity Data → Spatial Join → Surge Detection → XGBoost Forecast
       ↓                  ↓              ↓                  ↓
   5M records      Port→Terminal    Z-score based      24h/72h predictions
                   mapping          anomaly detection
```

## 🚀 Quick Start

```bash
# Clone
git clone https://github.com/anudeepadi/automatic-happiness.git
cd automatic-happiness

# Run on GPU (requires RAPIDS)
jupyter notebook port_to_rail_pipeline.ipynb
```

## 📁 Project Structure

```
├── port_to_rail_pipeline.ipynb   # Main integrated pipeline
├── PROJECT_STATUS.md             # Detailed documentation
├── data/                         # Data files (not in repo)
│   ├── Daily_Port_Activity_Data_and_Trade_Estimates.csv
│   ├── PortWatch_ports_database.csv
│   ├── NTAD_Rail_Network_Nodes.geojson
│   └── ...
└── output/                       # Results (not in repo)
```

## 🔧 Features

- **Spatial Join**: Match 114 US ports to nearest rail terminals
- **Drayage Estimation**: Calculate truck times and costs
- **Surge Detection**: Identify anomalies using rolling z-scores
- **GPU Acceleration**: cuDF, cuML, XGBoost with CUDA

## 📈 Results

| Model | MAE | R² |
|-------|-----|-----|
| XGBoost 24h | ~1.1 | ~0.7 |
| XGBoost 72h | ~1.2 | ~0.65 |

## 🛠️ Tech Stack

- **GPU**: RAPIDS cuDF, cuML
- **ML**: XGBoost (GPU)
- **Data**: IMF PortWatch, NTAD Rail Network

## 📝 Data Sources

- [IMF PortWatch](https://portwatch.imf.org/) - Daily port activity
- [NTAD Rail Network](https://geodata.bts.gov/) - US rail infrastructure
- Freight logistics data

## 👥 Team

DGX Spark Frontier Hackathon 2025

## 📄 License

MIT
