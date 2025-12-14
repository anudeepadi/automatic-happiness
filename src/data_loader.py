"""
Data loading and preprocessing for Port-to-Rail Surge Forecaster.
Optimized for GPU acceleration with RAPIDS cuDF.
"""

import json
import warnings
from pathlib import Path
from typing import Dict, Optional, Tuple, Union

import numpy as np
import pandas as pd

from .config import DATA_DIR, OUTPUT_DIR, DATA_CONFIG

warnings.filterwarnings('ignore')

# Try GPU acceleration
try:
    import cudf
    import cupy as cp
    GPU_AVAILABLE = True
    print(f"✅ GPU Mode: cuDF {cudf.__version__}")
except ImportError:
    cudf = pd
    cp = np
    GPU_AVAILABLE = False
    print("⚠️ CPU Mode: pandas")


DataFrame = Union[pd.DataFrame, "cudf.DataFrame"]


def load_port_activity(data_dir: Path = DATA_DIR) -> DataFrame:
    """Load IMF PortWatch daily port activity data."""
    filepath = data_dir / "Daily_Port_Activity_Data_and_Trade_Estimates.csv"

    print(f"Loading port activity from {filepath}...")

    if GPU_AVAILABLE:
        df = cudf.read_csv(str(filepath))
    else:
        df = pd.read_csv(filepath)

    print(f"✅ Loaded {len(df):,} records")
    return df


def load_port_database(data_dir: Path = DATA_DIR) -> DataFrame:
    """Load port metadata with coordinates."""
    filepath = data_dir / "PortWatch_ports_database.csv"

    if GPU_AVAILABLE:
        df = cudf.read_csv(str(filepath))
    else:
        df = pd.read_csv(filepath)

    print(f"✅ Port database: {len(df):,} ports")
    return df


def load_rail_nodes(data_dir: Path = DATA_DIR) -> DataFrame:
    """Load North American rail network nodes."""
    # Try different possible filenames
    patterns = [
        "NTAD_North_American_Rail_Network_Nodes*.geojson",
        "rail_nodes*.geojson",
    ]

    filepath = None
    for pattern in patterns:
        matches = list(data_dir.glob(pattern))
        if matches:
            filepath = matches[0]
            break

    if filepath is None:
        # Try loading from output if already processed
        output_file = OUTPUT_DIR / "rail_nodes_us.parquet"
        if output_file.exists():
            print(f"Loading preprocessed rail nodes from {output_file}...")
            if GPU_AVAILABLE:
                return cudf.read_parquet(str(output_file))
            return pd.read_parquet(output_file)
        raise FileNotFoundError("Rail nodes file not found")

    print(f"Loading rail nodes from {filepath}...")

    with open(filepath, 'r') as f:
        data = json.load(f)

    nodes = []
    for feat in data['features']:
        props = feat['properties']
        coords = feat['geometry']['coordinates']
        nodes.append({
            'node_id': props.get('FRANODEID'),
            'country': props.get('COUNTRY'),
            'state': props.get('STATE'),
            'fra_district': props.get('FRADISTRCT'),
            'node_lon': coords[0],
            'node_lat': coords[1]
        })

    df = pd.DataFrame(nodes)
    if GPU_AVAILABLE:
        df = cudf.DataFrame(df)

    print(f"✅ Rail nodes: {len(df):,} nodes")
    return df


def load_chokepoints(data_dir: Path = DATA_DIR) -> DataFrame:
    """Load port chokepoints data."""
    filepath = data_dir / "Daily_Chokepoints_Data.csv"

    if GPU_AVAILABLE:
        df = cudf.read_csv(str(filepath))
    else:
        df = pd.read_csv(filepath)

    print(f"✅ Chokepoints: {len(df):,} records")
    return df


def load_freight(data_dir: Path = DATA_DIR) -> DataFrame:
    """Load freight logistics data."""
    filepath = data_dir / "fFreight.csv"

    if GPU_AVAILABLE:
        df = cudf.read_csv(str(filepath))
    else:
        df = pd.read_csv(filepath)

    print(f"✅ Freight: {len(df):,} records")
    return df


def filter_us_data(
    port_activity: DataFrame,
    port_db: DataFrame,
    rail_nodes: DataFrame
) -> Tuple[DataFrame, DataFrame, DataFrame]:
    """Filter all datasets to US only."""

    print("Filtering to US data...")

    # US port activity
    ports_us = port_activity[port_activity['country'] == 'United States'].copy()
    print(f"  US Port Activity: {len(ports_us):,} records")

    # US port database
    ports_db_us = port_db[port_db['country'] == 'United States'].copy()
    print(f"  US Ports: {len(ports_db_us):,} ports")

    # US rail nodes
    rail_us = rail_nodes[rail_nodes['country'] == 'US'].copy()
    print(f"  US Rail Nodes: {len(rail_us):,} nodes")

    return ports_us, ports_db_us, rail_us


def parse_dates(df: DataFrame) -> DataFrame:
    """Parse date column to datetime."""
    if GPU_AVAILABLE:
        # cuDF date parsing
        df['date'] = df['date'].str.slice(0, 19)
        df['date'] = cudf.to_datetime(df['date'], format='%Y/%m/%d %H:%M:%S')
    else:
        df['date'] = pd.to_datetime(df['date'])

    return df


def haversine_distance(lat1, lon1, lat2, lon2) -> np.ndarray:
    """Calculate great-circle distance in km between points."""
    R = 6371  # Earth's radius in km

    lat1_rad = np.radians(lat1)
    lat2_rad = np.radians(lat2)
    dlat = np.radians(lat2 - lat1)
    dlon = np.radians(lon2 - lon1)

    a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
    c = 2 * np.arcsin(np.sqrt(a))

    return R * c


def create_port_terminal_mapping(
    port_db: DataFrame,
    rail_nodes: DataFrame,
    max_distance_km: float = DATA_CONFIG.max_drayage_distance_km
) -> pd.DataFrame:
    """Create spatial mapping from ports to nearest rail terminals."""

    print(f"Spatial join: matching ports to terminals (max {max_distance_km}km)...")

    # Convert to pandas for spatial ops
    if GPU_AVAILABLE:
        ports_pd = port_db.to_pandas()
        rail_pd = rail_nodes.to_pandas()
    else:
        ports_pd = port_db
        rail_pd = rail_nodes

    port_coords = ports_pd[['portid', 'portname', 'lat', 'lon']].copy()
    port_coords = port_coords.rename(columns={'lat': 'port_lat', 'lon': 'port_lon'})

    mappings = []
    for _, port in port_coords.iterrows():
        distances = haversine_distance(
            port['port_lat'], port['port_lon'],
            rail_pd['node_lat'].values, rail_pd['node_lon'].values
        )

        nearest_idx = np.argmin(distances)
        nearest_dist = distances[nearest_idx]

        if nearest_dist <= max_distance_km:
            nearest = rail_pd.iloc[nearest_idx]
            mappings.append({
                'portid': port['portid'],
                'portname': port['portname'],
                'port_lat': port['port_lat'],
                'port_lon': port['port_lon'],
                'terminal_id': nearest['node_id'],
                'terminal_state': nearest['state'],
                'terminal_lat': nearest['node_lat'],
                'terminal_lon': nearest['node_lon'],
                'distance_km': nearest_dist
            })

    df = pd.DataFrame(mappings)

    # Add drayage estimates
    df['drayage_time_min'] = (
        (df['distance_km'] / DATA_CONFIG.avg_truck_speed_kmh * 60) +
        DATA_CONFIG.loading_time_min + DATA_CONFIG.unloading_time_min
    )
    df['drayage_cost_usd'] = (
        df['distance_km'] * DATA_CONFIG.cost_per_km_usd +
        DATA_CONFIG.fixed_drayage_cost_usd
    )

    print(f"✅ Matched {len(df)} ports to terminals")
    print(f"   Avg distance: {df['distance_km'].mean():.1f} km")

    return df


def load_all_data(data_dir: Path = DATA_DIR) -> Dict[str, DataFrame]:
    """Load all datasets for the pipeline."""

    print("=" * 60)
    print("Loading Port-to-Rail datasets...")
    print("=" * 60)

    data = {}

    # Core datasets
    data['port_activity'] = load_port_activity(data_dir)
    data['port_database'] = load_port_database(data_dir)
    data['rail_nodes'] = load_rail_nodes(data_dir)

    # Optional datasets
    try:
        data['chokepoints'] = load_chokepoints(data_dir)
    except FileNotFoundError:
        print("⚠️ Chokepoints data not found")

    try:
        data['freight'] = load_freight(data_dir)
    except FileNotFoundError:
        print("⚠️ Freight data not found")

    # Filter to US
    data['ports_us'], data['ports_db_us'], data['rail_us'] = filter_us_data(
        data['port_activity'],
        data['port_database'],
        data['rail_nodes']
    )

    # Parse dates
    data['ports_us'] = parse_dates(data['ports_us'])

    # Create port-terminal mapping
    data['port_terminal_map'] = create_port_terminal_mapping(
        data['ports_db_us'],
        data['rail_us']
    )

    print("=" * 60)
    print("✅ All data loaded successfully")
    print("=" * 60)

    return data


def load_preprocessed_features(output_dir: Path = OUTPUT_DIR) -> Optional[DataFrame]:
    """Load preprocessed feature dataset if available."""

    filepath = output_dir / "champion_features.parquet"
    if not filepath.exists():
        filepath = output_dir / "silver_port_features.parquet"
    if not filepath.exists():
        filepath = output_dir / "port_features.parquet"

    if filepath.exists():
        print(f"Loading preprocessed features from {filepath}...")
        if GPU_AVAILABLE:
            return cudf.read_parquet(str(filepath))
        return pd.read_parquet(filepath)

    return None
