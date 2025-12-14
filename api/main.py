"""
FastAPI Backend for Port-to-Rail Surge Forecaster.
DGX Spark Frontier Hackathon 2025.

Endpoints:
- /health - System health check
- /predict - Run surge predictions
- /ports - Get port information
- /optimize - Get dispatch optimization
- /alerts - Get surge alerts
- /dashboard - Get dashboard data
"""

import sys
from datetime import datetime, timedelta
from pathlib import Path
from typing import Dict, List, Optional

import pandas as pd
from fastapi import FastAPI, HTTPException, Query
from fastapi.middleware.cors import CORSMiddleware
from pydantic import BaseModel

# Add src to path
sys.path.insert(0, str(Path(__file__).parent.parent))

from src.config import OUTPUT_DIR, MODEL_DIR
from src.model import ChampionModel

# Initialize FastAPI
app = FastAPI(
    title="Port-to-Rail Surge Forecaster",
    description="GPU-accelerated port surge prediction and rail dispatch optimization",
    version="1.0.0",
    docs_url="/docs",
    redoc_url="/redoc"
)

# CORS for frontend
app.add_middleware(
    CORSMiddleware,
    allow_origins=["*"],
    allow_credentials=True,
    allow_methods=["*"],
    allow_headers=["*"],
)

# Global state
_data_cache: Dict = {}
_last_load: Optional[datetime] = None
_model: Optional[ChampionModel] = None


def load_model():
    """Load the trained ChampionModel."""
    global _model

    if _model is not None:
        return _model

    try:
        _model = ChampionModel.load(MODEL_DIR, prefix='champion')
        print(f"✅ ChampionModel loaded with targets: {list(_model.models.keys())}")
        return _model
    except Exception as e:
        print(f"⚠️ Could not load model: {e}")
        return None


def load_data():
    """Load preprocessed data."""
    global _data_cache, _last_load

    # Reload every 5 minutes
    if _last_load and (datetime.now() - _last_load).seconds < 300:
        return _data_cache

    print("Loading data...")

    # Load port-terminal mapping
    mapping_path = OUTPUT_DIR / "port_terminal_mapping.csv"
    if mapping_path.exists():
        _data_cache['port_terminal_map'] = pd.read_csv(mapping_path)

    # Load surge analysis
    surge_path = OUTPUT_DIR / "surge_analysis.csv"
    if surge_path.exists():
        _data_cache['surge_analysis'] = pd.read_csv(surge_path)

    # Load unified freight model
    freight_path = OUTPUT_DIR / "unified_freight_model.parquet"
    if freight_path.exists():
        _data_cache['freight_model'] = pd.read_parquet(freight_path)
    else:
        # Fallback to port features
        features_path = OUTPUT_DIR / "port_features.parquet"
        if features_path.exists():
            _data_cache['freight_model'] = pd.read_parquet(features_path)

    # Load feature importance
    imp_path = OUTPUT_DIR / "feature_importance.csv"
    if imp_path.exists():
        _data_cache['feature_importance'] = pd.read_csv(imp_path)

    # Load champion features (for model inference)
    features_path = OUTPUT_DIR / "champion_features.parquet"
    if features_path.exists():
        _data_cache['champion_features'] = pd.read_parquet(features_path)

    _last_load = datetime.now()
    print(f"✅ Data loaded: {list(_data_cache.keys())}")

    return _data_cache


# Pydantic models
class HealthResponse(BaseModel):
    status: str
    timestamp: str
    data_loaded: bool
    datasets: List[str]


class PortInfo(BaseModel):
    portid: str
    portname: str
    port_lat: float
    port_lon: float
    terminal_id: Optional[int]
    terminal_state: Optional[str]
    distance_km: float
    drayage_time_min: float
    drayage_cost_usd: float


class SurgeAnalysis(BaseModel):
    port: str
    total_days: int
    surge_2std_days: int
    surge_rate: float
    avg_calls: float
    max_calls: float
    max_zscore: float
    avg_import: float


class PredictionResponse(BaseModel):
    port: str
    date: str
    predicted_calls_1d: float
    predicted_calls_3d: float
    surge_probability: float
    risk_level: str
    confidence: float


class DispatchWindow(BaseModel):
    port: str
    state: str
    optimal_hours: str
    risk_score: float
    expected_trucks: int
    recommendation: str


class RepositioningRec(BaseModel):
    from_terminal: str
    to_terminal: str
    trucks: int
    distance_km: float
    urgency: str
    reason: str


class OptimizationResponse(BaseModel):
    date: str
    summary: Dict
    dispatch_windows: List[DispatchWindow]
    repositioning: List[RepositioningRec]
    terminal_status: List[Dict]


class DashboardData(BaseModel):
    summary: Dict
    ports: List[Dict]
    surge_analysis: List[Dict]
    predictions: List[Dict]
    optimization: Dict
    feature_importance: List[Dict]


# Routes
@app.get("/", tags=["Root"])
async def root():
    """API root."""
    return {
        "name": "Port-to-Rail Surge Forecaster API",
        "version": "1.0.0",
        "docs": "/docs",
        "health": "/health"
    }


@app.get("/health", response_model=HealthResponse, tags=["System"])
async def health_check():
    """System health check."""
    data = load_data()
    return HealthResponse(
        status="healthy",
        timestamp=datetime.now().isoformat(),
        data_loaded=len(data) > 0,
        datasets=list(data.keys())
    )


@app.get("/ports", response_model=List[PortInfo], tags=["Ports"])
async def get_ports(
    limit: int = Query(50, ge=1, le=200),
    state: Optional[str] = Query(None, description="Filter by state")
):
    """Get port information with terminal mapping."""
    data = load_data()

    if 'port_terminal_map' not in data:
        raise HTTPException(status_code=404, detail="Port data not found")

    df = data['port_terminal_map'].copy()

    if state:
        df = df[df['terminal_state'] == state]

    df = df.head(limit)

    return df.to_dict('records')


@app.get("/ports/{port_name}", tags=["Ports"])
async def get_port(port_name: str):
    """Get specific port information."""
    data = load_data()

    if 'port_terminal_map' not in data:
        raise HTTPException(status_code=404, detail="Port data not found")

    df = data['port_terminal_map']
    port = df[df['portname'].str.lower() == port_name.lower()]

    if port.empty:
        raise HTTPException(status_code=404, detail=f"Port '{port_name}' not found")

    result = port.iloc[0].to_dict()

    # Add surge analysis if available
    if 'surge_analysis' in data:
        surge = data['surge_analysis']
        port_surge = surge[surge['port'].str.lower() == port_name.lower()]
        if not port_surge.empty:
            result['surge_analysis'] = port_surge.iloc[0].to_dict()

    return result


@app.get("/surge-analysis", response_model=List[SurgeAnalysis], tags=["Analysis"])
async def get_surge_analysis(
    limit: int = Query(30, ge=1, le=100),
    min_surge_rate: float = Query(0, ge=0, le=100)
):
    """Get surge analysis for all ports."""
    data = load_data()

    if 'surge_analysis' not in data:
        raise HTTPException(status_code=404, detail="Surge analysis not found")

    df = data['surge_analysis'].copy()

    if min_surge_rate > 0:
        df = df[df['surge_rate'] >= min_surge_rate]

    df = df.head(limit)

    return df.to_dict('records')


@app.get("/predictions", tags=["Predictions"])
async def get_predictions(
    port: Optional[str] = Query(None, description="Filter by port name"),
    days_ahead: int = Query(1, ge=1, le=7, description="Forecast horizon"),
    limit: int = Query(50, ge=1, le=200)
):
    """Get port activity predictions using trained XGBoost model."""
    data = load_data()
    model = load_model()

    # Try to use champion features for model inference
    if 'champion_features' in data:
        df = data['champion_features'].copy()
    elif 'freight_model' in data:
        df = data['freight_model'].copy()
    else:
        raise HTTPException(status_code=404, detail="Prediction data not found")

    # Handle port filter (check it's a valid string, not Query object)
    if port and isinstance(port, str):
        df = df[df['portname'].str.lower() == port.lower()]

    # Get latest data per port
    df = df.sort_values('date', ascending=False)

    predictions = []
    for port_name in df['portname'].unique()[:limit]:
        port_data = df[df['portname'] == port_name].head(1)
        if port_data.empty:
            continue

        row = port_data.iloc[0]

        # Try model predictions first
        if model is not None and hasattr(model, 'feature_cols') and len(model.feature_cols) > 0:
            try:
                # Prepare features for model
                feature_row = port_data[model.feature_cols].fillna(0)

                # Get predictions from trained model
                pred_1d = float(model.predict(feature_row, 'calls_1d')[0]) if 'calls_1d' in model.models else 0
                pred_3d = float(model.predict(feature_row, 'calls_3d')[0]) if 'calls_3d' in model.models else 0

                # Surge prediction (binary classification)
                if 'surge_1d' in model.models:
                    surge_prob = float(model.predict(feature_row, 'surge_1d')[0]) * 100
                else:
                    # Fallback: estimate from zscore
                    zscore = row.get('zscore_7d', 0)
                    surge_prob = min(max((zscore + 2) / 4, 0), 1) * 100

                risk_level = 'HIGH' if surge_prob > 70 else 'MEDIUM' if surge_prob > 40 else 'LOW'

                predictions.append({
                    'port': port_name,
                    'date': datetime.now().isoformat(),
                    'predicted_calls_1d': round(pred_1d, 2),
                    'predicted_calls_3d': round(pred_3d, 2),
                    'surge_probability': round(surge_prob, 1),
                    'risk_level': risk_level,
                    'confidence': round(model.metrics.get('calls_1d', {}).get('r2', 0.5) * 100, 1),
                    'model_used': 'XGBoost'
                })
                continue
            except Exception as e:
                print(f"Model prediction failed for {port_name}: {e}")

        # Fallback: heuristic-based prediction
        ma7 = row.get('ma7', row.get('portcalls', 5))
        zscore = row.get('zscore_7d', row.get('zscore', 0))
        surge_prob = min(max((zscore + 2) / 4, 0), 1)

        risk_level = 'HIGH' if surge_prob > 0.7 else 'MEDIUM' if surge_prob > 0.4 else 'LOW'

        predictions.append({
            'port': port_name,
            'date': datetime.now().isoformat(),
            'predicted_calls_1d': round(float(ma7 * (1 + zscore * 0.1)), 2),
            'predicted_calls_3d': round(float(ma7 * (1 + zscore * 0.05)), 2),
            'surge_probability': round(surge_prob * 100, 1),
            'risk_level': risk_level,
            'confidence': round(85 - abs(zscore) * 5, 1),
            'model_used': 'Heuristic'
        })

    return predictions


@app.get("/optimize", response_model=OptimizationResponse, tags=["Optimization"])
async def get_optimization():
    """Get dispatch optimization recommendations."""
    data = load_data()

    if 'port_terminal_map' not in data:
        raise HTTPException(status_code=404, detail="Port data not found")

    port_map = data['port_terminal_map']
    freight = data.get('freight_model', pd.DataFrame())

    # Generate optimization recommendations
    dispatch_windows = []
    for _, row in port_map.iterrows():
        # Get port surge data
        port_surge = 0.3
        if 'surge_analysis' in data:
            surge_df = data['surge_analysis']
            port_row = surge_df[surge_df['port'] == row['portname']]
            if not port_row.empty:
                port_surge = port_row.iloc[0]['surge_rate'] / 100

        risk_score = min(port_surge * 2, 1.0)

        if risk_score < 0.3:
            hours = "06:00 - 22:00"
            rec = "Standard dispatch - low congestion expected"
        elif risk_score < 0.6:
            hours = "10:00 - 15:00"
            rec = "Off-peak dispatch recommended"
        else:
            hours = "22:00 - 06:00"
            rec = "DELAY or overnight dispatch - high surge risk"

        dispatch_windows.append(DispatchWindow(
            port=row['portname'],
            state=row['terminal_state'],
            optimal_hours=hours,
            risk_score=round(risk_score, 2),
            expected_trucks=int(row.get('daily_trucks_needed', 100)),
            recommendation=rec
        ))

    # Sort by risk
    dispatch_windows.sort(key=lambda x: -x.risk_score)

    # Generate repositioning recommendations
    repositioning = []
    sorted_ports = sorted(dispatch_windows, key=lambda x: x.risk_score)

    for i in range(min(5, len(sorted_ports) // 2)):
        low_risk = sorted_ports[i]
        high_risk = sorted_ports[-(i+1)]

        if high_risk.risk_score > 0.5 and low_risk.risk_score < 0.3:
            repositioning.append(RepositioningRec(
                from_terminal=low_risk.port,
                to_terminal=high_risk.port,
                trucks=min(50, high_risk.expected_trucks // 4),
                distance_km=100.0,
                urgency='high' if high_risk.risk_score > 0.7 else 'medium',
                reason=f"Rebalance for predicted surge at {high_risk.port}"
            ))

    # Terminal status
    terminal_status = []
    for _, row in port_map.head(20).iterrows():
        utilization = min(row.get('daily_trucks_needed', 100) / 500 * 100, 100)
        status = 'CRITICAL' if utilization > 90 else 'HIGH' if utilization > 75 else 'NORMAL' if utilization > 50 else 'LOW'

        terminal_status.append({
            'port': row['portname'],
            'state': row['terminal_state'],
            'utilization_pct': round(utilization, 1),
            'status': status,
            'distance_km': round(row['distance_km'], 1),
            'drayage_time_min': round(row['drayage_time_min'], 0)
        })

    return OptimizationResponse(
        date=datetime.now().isoformat(),
        summary={
            'total_ports': len(port_map),
            'high_risk_ports': len([w for w in dispatch_windows if w.risk_score > 0.6]),
            'critical_terminals': len([t for t in terminal_status if t['status'] == 'CRITICAL']),
            'total_trucks_to_reposition': sum(r.trucks for r in repositioning),
            'avg_utilization_pct': round(sum(t['utilization_pct'] for t in terminal_status) / max(len(terminal_status), 1), 1)
        },
        dispatch_windows=dispatch_windows[:10],
        repositioning=repositioning,
        terminal_status=terminal_status
    )


@app.get("/alerts", tags=["Alerts"])
async def get_alerts(
    threshold: float = Query(50, ge=0, le=100, description="Surge probability threshold")
):
    """Get active surge alerts."""
    predictions = await get_predictions(limit=100)

    alerts = [
        p for p in predictions
        if p['surge_probability'] >= threshold
    ]

    return sorted(alerts, key=lambda x: -x['surge_probability'])


@app.get("/feature-importance", tags=["Analysis"])
async def get_feature_importance(limit: int = Query(20, ge=1, le=50)):
    """Get model feature importance."""
    data = load_data()

    if 'feature_importance' not in data:
        raise HTTPException(status_code=404, detail="Feature importance not found")

    df = data['feature_importance'].head(limit)
    return df.to_dict('records')


@app.get("/dashboard", response_model=DashboardData, tags=["Dashboard"])
async def get_dashboard_data():
    """Get all data for dashboard rendering."""
    data = load_data()

    # Summary stats
    port_map = data.get('port_terminal_map', pd.DataFrame())
    surge_analysis = data.get('surge_analysis', pd.DataFrame())
    feature_imp = data.get('feature_importance', pd.DataFrame())

    summary = {
        'total_ports': len(port_map),
        'total_terminals': len(port_map),
        'avg_drayage_distance_km': round(port_map['distance_km'].mean(), 1) if not port_map.empty else 0,
        'avg_drayage_time_min': round(port_map['drayage_time_min'].mean(), 0) if not port_map.empty else 0,
        'avg_drayage_cost_usd': round(port_map['drayage_cost_usd'].mean(), 0) if not port_map.empty else 0,
        'high_surge_ports': len(surge_analysis[surge_analysis['surge_rate'] > 0.8]) if not surge_analysis.empty else 0,
        'data_date_range': '2019-01-01 to 2025-11-28',
        'last_updated': datetime.now().isoformat()
    }

    # Get predictions
    predictions = await get_predictions(port=None, days_ahead=1, limit=30)

    # Get optimization
    optimization = await get_optimization()

    return DashboardData(
        summary=summary,
        ports=port_map.head(50).to_dict('records') if not port_map.empty else [],
        surge_analysis=surge_analysis.head(30).to_dict('records') if not surge_analysis.empty else [],
        predictions=predictions,
        optimization=optimization.dict(),
        feature_importance=feature_imp.head(20).to_dict('records') if not feature_imp.empty else []
    )


@app.get("/stats", tags=["Analysis"])
async def get_stats():
    """Get aggregate statistics."""
    data = load_data()

    stats = {}

    if 'port_terminal_map' in data:
        df = data['port_terminal_map']
        stats['ports'] = {
            'total': len(df),
            'states': df['terminal_state'].nunique(),
            'avg_distance_km': round(df['distance_km'].mean(), 2),
            'max_distance_km': round(df['distance_km'].max(), 2),
            'avg_drayage_min': round(df['drayage_time_min'].mean(), 1),
            'avg_cost_usd': round(df['drayage_cost_usd'].mean(), 2),
            'total_daily_trucks': int(df['daily_trucks_needed'].sum()) if 'daily_trucks_needed' in df.columns else 0
        }

    if 'surge_analysis' in data:
        df = data['surge_analysis']
        stats['surges'] = {
            'ports_analyzed': len(df),
            'avg_surge_rate': round(df['surge_rate'].mean(), 2),
            'max_surge_rate': round(df['surge_rate'].max(), 2),
            'avg_daily_calls': round(df['avg_calls'].mean(), 1),
            'max_daily_calls': round(df['max_calls'].max(), 1)
        }

    if 'freight_model' in data:
        df = data['freight_model']
        stats['activity'] = {
            'total_records': len(df),
            'unique_ports': df['portname'].nunique() if 'portname' in df.columns else 0,
            'date_range': {
                'start': str(df['date'].min()) if 'date' in df.columns else None,
                'end': str(df['date'].max()) if 'date' in df.columns else None
            }
        }

    return stats


# Startup event
@app.on_event("startup")
async def startup_event():
    """Load data and model on startup."""
    print("=" * 60)
    print("🚢 Port-to-Rail Surge Forecaster API")
    print("=" * 60)
    load_data()
    model = load_model()
    if model:
        print(f"✅ XGBoost model loaded: {list(model.models.keys())}")
    else:
        print("⚠️ Running in heuristic mode (no trained model)")
    print("✅ API ready")


if __name__ == "__main__":
    import uvicorn
    uvicorn.run(app, host="0.0.0.0", port=8000)
