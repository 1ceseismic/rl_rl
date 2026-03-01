import numpy as np
from typing import Any, Dict, List

from rlgym.api import AgentID, SharedInfoProvider
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import CAR_MAX_SPEED
from rlgym_learn_algos.ppo import PPOMetricsLogger


class CustomMetricsProvider(SharedInfoProvider[AgentID, GameState]):
    """Captures per-step game state and accumulates stats for metrics logging."""

    def create(self, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        shared_info["game_metrics"] = self._empty_metrics()
        shared_info["_step_count"] = 0
        shared_info["_total_touches"] = 0
        shared_info["_demo_count"] = 0
        return shared_info

    def set_state(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        shared_info["game_metrics"] = self._empty_metrics()
        shared_info["_step_count"] = 0
        shared_info["_total_touches"] = 0
        shared_info["_demo_count"] = 0
        return shared_info

    def step(
        self,
        agents: List[AgentID],
        state: GameState,
        shared_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        metrics = shared_info["game_metrics"]
        shared_info["_step_count"] += 1
        n = shared_info["_step_count"]

        ball_pos = state.ball.position
        n_cars = len(state.cars)

        total_boost = 0.0
        total_speed = 0.0
        total_dist_to_ball = 0.0
        airborne_count = 0
        supersonic_count = 0
        demo_count = 0
        total_touches = 0

        for agent_id, car in state.cars.items():
            total_boost += car.boost_amount
            speed = np.linalg.norm(car.physics.linear_velocity)
            total_speed += speed
            total_dist_to_ball += np.linalg.norm(car.physics.position - ball_pos)
            if not car.on_ground:
                airborne_count += 1
            if car.is_supersonic:
                supersonic_count += 1
            if car.is_demoed:
                demo_count += 1
            total_touches += car.ball_touches

        if n_cars > 0:
            # Running averages using incremental formula
            avg_boost = total_boost / n_cars
            avg_speed = total_speed / n_cars
            avg_dist = total_dist_to_ball / n_cars
            air_ratio = airborne_count / n_cars
            supersonic_ratio = supersonic_count / n_cars

            metrics["Avg Boost"] += (avg_boost - metrics["Avg Boost"]) / n
            metrics["Avg Speed"] += (avg_speed - metrics["Avg Speed"]) / n
            metrics["Avg Distance to Ball"] += (avg_dist - metrics["Avg Distance to Ball"]) / n
            metrics["Air Time Ratio"] += (air_ratio - metrics["Air Time Ratio"]) / n
            metrics["Supersonic Ratio"] += (supersonic_ratio - metrics["Supersonic Ratio"]) / n

        # Track new demos and touches since last step
        new_demos = demo_count - shared_info.get("_prev_demo_count", 0)
        if new_demos > 0:
            shared_info["_demo_count"] += new_demos
        shared_info["_prev_demo_count"] = demo_count

        shared_info["_total_touches"] = total_touches
        metrics["Ball Touches"] = float(total_touches)
        metrics["Demos"] = float(shared_info["_demo_count"])

        if state.goal_scored:
            if state.scoring_team == 0:
                metrics["Blue Goals"] += 1.0
            else:
                metrics["Orange Goals"] += 1.0

        shared_info["game_metrics"] = metrics
        return shared_info

    @staticmethod
    def _empty_metrics() -> Dict[str, float]:
        return {
            "Blue Goals": 0.0,
            "Orange Goals": 0.0,
            "Avg Boost": 0.0,
            "Avg Speed": 0.0,
            "Air Time Ratio": 0.0,
            "Ball Touches": 0.0,
            "Supersonic Ratio": 0.0,
            "Avg Distance to Ball": 0.0,
            "Demos": 0.0,
        }


class CustomMetricsLogger(PPOMetricsLogger):
    """Extends PPOMetricsLogger to aggregate game-specific metrics from environments."""

    def collect_env_metrics(self, data: List[Dict[str, Any]]):
        if not data:
            self.state_metrics = {}
            return

        # Filter to only entries that have game_metrics
        game_data = [d["game_metrics"] for d in data if isinstance(d, dict) and "game_metrics" in d]
        if not game_data:
            self.state_metrics = {}
            return

        # Aggregate by averaging across all env snapshots
        keys = game_data[0].keys()
        aggregated = {}
        for key in keys:
            values = [d[key] for d in game_data]
            if "Goals" in key or "Demos" in key or "Touches" in key:
                aggregated[key] = sum(values) / len(values)
            else:
                aggregated[key] = sum(values) / len(values)

        # Normalize speed to a more readable value
        if aggregated.get("Avg Speed", 0) > 0:
            aggregated["Avg Speed (% Max)"] = aggregated.pop("Avg Speed") / CAR_MAX_SPEED * 100

        self.state_metrics = {"Game Stats": aggregated}
