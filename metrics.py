import numpy as np
from typing import Any, Dict, List

from rlgym.api import AgentID, SharedInfoProvider
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import CAR_MAX_SPEED
from rlgym_learn_algos.ppo import PPOMetricsLogger

from engine import rsim_stats


class CustomMetricsProvider(SharedInfoProvider[AgentID, GameState]):
    """Captures per-step game state and accumulates stats for metrics logging."""

    def create(self, shared_info: Dict[str, Any]) -> Dict[str, Any]:
        shared_info["game_metrics"] = self._empty_metrics()
        shared_info["_step_count"] = 0
        shared_info["_prev_flipping"] = set()
        return shared_info

    def set_state(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> Dict[str, Any]:
        shared_info["game_metrics"] = self._empty_metrics()
        shared_info["_step_count"] = 0
        shared_info["_prev_flipping"] = set()
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
        total_touches = 0
        prev_flipping = shared_info["_prev_flipping"]
        now_flipping = set()

        for agent_id, car in state.cars.items():
            total_boost += car.boost_amount
            speed = np.linalg.norm(car.physics.linear_velocity)
            total_speed += speed
            total_dist_to_ball += np.linalg.norm(car.physics.position - ball_pos)
            if not car.on_ground:
                airborne_count += 1
            if car.is_supersonic:
                supersonic_count += 1
            total_touches += car.ball_touches

            # Count flip starts (transition from not flipping to flipping)
            if car.is_flipping:
                now_flipping.add(agent_id)
                if agent_id not in prev_flipping:
                    metrics["Flips"] += 1.0

            # Accumulate rsim stats (saves, shots, assists, demos, boost pickups)
            agent_stats = rsim_stats.get(agent_id)
            if agent_stats:
                metrics["Demos"] += agent_stats["demos"]
                metrics["Saves"] += agent_stats["saves"]
                metrics["Shots"] += agent_stats["shots"]
                metrics["Assists"] += agent_stats["assists"]
                metrics["Boost Pickups"] += agent_stats["boost_pickups"]

        shared_info["_prev_flipping"] = now_flipping

        if n_cars > 0:
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

        metrics["Ball Touches"] += total_touches

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
            "Saves": 0.0,
            "Shots": 0.0,
            "Assists": 0.0,
            "Flips": 0.0,
            "Boost Pickups": 0.0,
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
            aggregated[key] = sum(values) / len(values)

        # Normalize speed to a more readable value
        if aggregated.get("Avg Speed", 0) > 0:
            aggregated["Avg Speed (% Max)"] = aggregated.pop("Avg Speed") / CAR_MAX_SPEED * 100

        # Per-component reward breakdown
        reward_data = [d["reward_components"] for d in data if isinstance(d, dict) and "reward_components" in d]
        reward_breakdown = {}
        if reward_data:
            for key in reward_data[0]:
                values = [d[key] for d in reward_data]
                reward_breakdown[key] = sum(values) / len(values)

        self.state_metrics = {"Game Stats": aggregated, "Reward Components": reward_breakdown}
