"""
Rail dispatch optimization and terminal utilization module.
Implements the missing challenge requirements:
- Ideal time windows for rail dispatch
- Terminal repositioning recommendations
- Rail utilization scoring
"""

from dataclasses import dataclass
from datetime import datetime, timedelta
from typing import Dict, List, Optional, Tuple

import numpy as np
import pandas as pd


@dataclass
class DispatchWindow:
    """Recommended dispatch window."""
    port_name: str
    terminal_state: str
    start_hour: int
    end_hour: int
    risk_score: float  # 0-1, lower is better
    expected_trucks: int
    recommended_action: str


@dataclass
class RepositioningRecommendation:
    """Truck repositioning recommendation."""
    from_terminal: str
    to_terminal: str
    n_trucks: int
    distance_km: float
    reason: str
    urgency: str  # 'high', 'medium', 'low'


class DispatchOptimizer:
    """
    Optimizes rail dispatch timing and truck positioning.

    Key features:
    - Identifies optimal dispatch windows based on surge predictions
    - Recommends truck repositioning between terminals
    - Calculates terminal utilization scores
    """

    def __init__(
        self,
        port_terminal_map: pd.DataFrame,
        surge_predictions: Optional[pd.DataFrame] = None
    ):
        self.port_terminal_map = port_terminal_map
        self.surge_predictions = surge_predictions

        # Default capacity assumptions
        self.terminal_capacity = 500  # trucks per day
        self.optimal_utilization = 0.75  # target 75%

    def calculate_dispatch_windows(
        self,
        predictions: pd.DataFrame,
        date: datetime
    ) -> List[DispatchWindow]:
        """
        Calculate optimal dispatch windows for each port.

        Uses predictions to identify low-risk periods for dispatch.
        """

        windows = []

        # Get unique ports
        ports = predictions['portname'].unique() if 'portname' in predictions.columns else []

        for port in ports:
            port_data = predictions[predictions['portname'] == port]

            if port_data.empty:
                continue

            # Get terminal info
            terminal_info = self.port_terminal_map[
                self.port_terminal_map['portname'] == port
            ]

            if terminal_info.empty:
                continue

            terminal_state = terminal_info.iloc[0]['terminal_state']

            # Calculate risk score based on surge probability
            surge_prob = port_data.get('pred_surge_1d', port_data.get('surge_2std', 0))
            if isinstance(surge_prob, pd.Series):
                surge_prob = surge_prob.mean()

            # Lower surge = better dispatch window
            risk_score = float(surge_prob) if not pd.isna(surge_prob) else 0.5

            # Determine optimal hours (avoid rush hours 6-9am and 4-7pm)
            if risk_score < 0.3:
                # Low risk: flexible timing
                start_hour, end_hour = 6, 22
                action = "Standard dispatch - low congestion expected"
            elif risk_score < 0.6:
                # Medium risk: prefer off-peak
                start_hour, end_hour = 10, 15
                action = "Off-peak dispatch recommended - moderate activity"
            else:
                # High risk: minimize dispatch
                start_hour, end_hour = 22, 6  # overnight
                action = "DELAY or overnight dispatch - high surge risk"

            # Estimate truck demand
            avg_import = terminal_info.iloc[0].get('avg_import_teu', 1000)
            expected_trucks = int(avg_import * 0.6 / 2)  # 60% rail, 2 TEU/truck

            windows.append(DispatchWindow(
                port_name=port,
                terminal_state=terminal_state,
                start_hour=start_hour,
                end_hour=end_hour,
                risk_score=risk_score,
                expected_trucks=expected_trucks,
                recommended_action=action
            ))

        return sorted(windows, key=lambda x: x.risk_score)

    def calculate_repositioning(
        self,
        predictions: pd.DataFrame,
        current_inventory: Optional[Dict[str, int]] = None
    ) -> List[RepositioningRecommendation]:
        """
        Recommend truck repositioning between terminals.

        Balances truck inventory based on predicted demand.
        """

        recommendations = []

        if current_inventory is None:
            # Assume even distribution
            n_terminals = len(self.port_terminal_map)
            trucks_per_terminal = 100
            current_inventory = {
                row['portname']: trucks_per_terminal
                for _, row in self.port_terminal_map.iterrows()
            }

        # Calculate demand imbalance
        port_demand = {}
        for port in current_inventory.keys():
            if 'portname' in predictions.columns:
                port_data = predictions[predictions['portname'] == port]
                if not port_data.empty:
                    pred_calls = port_data.get('pred_calls_1d', port_data.get('portcalls', 5))
                    if isinstance(pred_calls, pd.Series):
                        pred_calls = pred_calls.mean()
                    port_demand[port] = float(pred_calls) * 20  # ~20 trucks per ship call
                else:
                    port_demand[port] = 50
            else:
                port_demand[port] = 50

        # Find surplus and deficit ports
        surplus = []
        deficit = []

        for port, inventory in current_inventory.items():
            demand = port_demand.get(port, 50)
            balance = inventory - demand

            if balance > 20:
                surplus.append((port, balance))
            elif balance < -20:
                deficit.append((port, -balance))

        surplus.sort(key=lambda x: -x[1])  # Most surplus first
        deficit.sort(key=lambda x: -x[1])  # Most deficit first

        # Match surplus to deficit (greedy)
        for def_port, def_amount in deficit:
            for i, (sur_port, sur_amount) in enumerate(surplus):
                if sur_amount <= 0:
                    continue

                # Calculate distance
                sur_info = self.port_terminal_map[
                    self.port_terminal_map['portname'] == sur_port
                ]
                def_info = self.port_terminal_map[
                    self.port_terminal_map['portname'] == def_port
                ]

                if sur_info.empty or def_info.empty:
                    continue

                distance = self._haversine(
                    sur_info.iloc[0]['port_lat'],
                    sur_info.iloc[0]['port_lon'],
                    def_info.iloc[0]['port_lat'],
                    def_info.iloc[0]['port_lon']
                )

                # Skip if too far (>500km)
                if distance > 500:
                    continue

                transfer_amount = min(sur_amount, def_amount)

                # Determine urgency
                if def_amount > 50:
                    urgency = 'high'
                elif def_amount > 20:
                    urgency = 'medium'
                else:
                    urgency = 'low'

                recommendations.append(RepositioningRecommendation(
                    from_terminal=sur_port,
                    to_terminal=def_port,
                    n_trucks=int(transfer_amount),
                    distance_km=round(distance, 1),
                    reason=f"Predicted demand surge at {def_port}",
                    urgency=urgency
                ))

                surplus[i] = (sur_port, sur_amount - transfer_amount)
                def_amount -= transfer_amount

                if def_amount <= 0:
                    break

        return sorted(recommendations, key=lambda x: {'high': 0, 'medium': 1, 'low': 2}[x.urgency])

    def calculate_terminal_utilization(
        self,
        predictions: pd.DataFrame,
        date: datetime
    ) -> pd.DataFrame:
        """Calculate utilization scores for each terminal."""

        results = []

        for _, row in self.port_terminal_map.iterrows():
            port = row['portname']

            # Get predictions
            if 'portname' in predictions.columns:
                port_preds = predictions[predictions['portname'] == port]
            else:
                port_preds = pd.DataFrame()

            # Calculate metrics
            if not port_preds.empty:
                pred_calls = port_preds.get('pred_calls_1d', port_preds.get('portcalls', 5))
                if isinstance(pred_calls, pd.Series):
                    pred_calls = pred_calls.mean()

                surge_prob = port_preds.get('pred_surge_1d', port_preds.get('surge_2std', 0))
                if isinstance(surge_prob, pd.Series):
                    surge_prob = surge_prob.mean()
            else:
                pred_calls = 5
                surge_prob = 0.3

            # Estimate utilization
            expected_trucks = float(pred_calls) * 20
            utilization = expected_trucks / self.terminal_capacity

            # Status
            if utilization > 0.9:
                status = 'CRITICAL'
                color = 'red'
            elif utilization > 0.75:
                status = 'HIGH'
                color = 'orange'
            elif utilization > 0.5:
                status = 'NORMAL'
                color = 'green'
            else:
                status = 'LOW'
                color = 'blue'

            results.append({
                'port': port,
                'terminal_state': row['terminal_state'],
                'predicted_calls': round(float(pred_calls), 1),
                'expected_trucks': int(expected_trucks),
                'utilization_pct': round(utilization * 100, 1),
                'surge_probability': round(float(surge_prob) * 100, 1),
                'status': status,
                'status_color': color,
                'distance_km': round(row['distance_km'], 1),
                'drayage_time_min': round(row['drayage_time_min'], 0),
                'drayage_cost_usd': round(row['drayage_cost_usd'], 0),
            })

        return pd.DataFrame(results).sort_values('utilization_pct', ascending=False)

    def generate_daily_plan(
        self,
        predictions: pd.DataFrame,
        date: datetime
    ) -> Dict:
        """Generate comprehensive daily dispatch plan."""

        dispatch_windows = self.calculate_dispatch_windows(predictions, date)
        repositioning = self.calculate_repositioning(predictions)
        utilization = self.calculate_terminal_utilization(predictions, date)

        # Summary statistics
        n_high_risk = len([w for w in dispatch_windows if w.risk_score > 0.6])
        n_critical_terminals = len(utilization[utilization['status'] == 'CRITICAL'])
        total_repositioning = sum(r.n_trucks for r in repositioning)

        return {
            'date': date.isoformat(),
            'summary': {
                'total_ports': len(dispatch_windows),
                'high_risk_ports': n_high_risk,
                'critical_terminals': n_critical_terminals,
                'total_trucks_to_reposition': total_repositioning,
                'avg_utilization_pct': round(utilization['utilization_pct'].mean(), 1),
            },
            'dispatch_windows': [
                {
                    'port': w.port_name,
                    'state': w.terminal_state,
                    'optimal_hours': f"{w.start_hour:02d}:00 - {w.end_hour:02d}:00",
                    'risk_score': round(w.risk_score, 2),
                    'expected_trucks': w.expected_trucks,
                    'recommendation': w.recommended_action
                }
                for w in dispatch_windows[:10]  # Top 10
            ],
            'repositioning': [
                {
                    'from': r.from_terminal,
                    'to': r.to_terminal,
                    'trucks': r.n_trucks,
                    'distance_km': r.distance_km,
                    'urgency': r.urgency,
                    'reason': r.reason
                }
                for r in repositioning[:10]
            ],
            'terminal_status': utilization.head(20).to_dict('records')
        }

    @staticmethod
    def _haversine(lat1: float, lon1: float, lat2: float, lon2: float) -> float:
        """Calculate distance in km between two points."""
        R = 6371
        lat1_rad = np.radians(lat1)
        lat2_rad = np.radians(lat2)
        dlat = np.radians(lat2 - lat1)
        dlon = np.radians(lon2 - lon1)

        a = np.sin(dlat/2)**2 + np.cos(lat1_rad) * np.cos(lat2_rad) * np.sin(dlon/2)**2
        c = 2 * np.arcsin(np.sqrt(a))
        return R * c


def optimize_dispatch(
    port_terminal_map: pd.DataFrame,
    predictions: pd.DataFrame,
    date: Optional[datetime] = None
) -> Dict:
    """
    Main optimization function.

    Returns comprehensive dispatch recommendations.
    """

    if date is None:
        date = datetime.now()

    optimizer = DispatchOptimizer(port_terminal_map, predictions)
    return optimizer.generate_daily_plan(predictions, date)
