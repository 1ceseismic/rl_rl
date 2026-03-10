from rlgym.rocket_league.sim import RocketSimEngine

# Module-level dict so the metrics provider can read it without a reference to the engine.
# Keyed by agent_id, values are dicts of cumulative rsim stats.
rsim_stats = {}


class StatsEngine(RocketSimEngine):
    """RocketSimEngine that also exposes per-car rsim stats (saves, shots, etc.)."""

    def __init__(self, **kwargs):
        super().__init__(**kwargs)
        self._prev_stats = {}

    def _get_state(self):
        gs = super()._get_state()

        for agent_id, rsim_car in self._cars.items():
            current = {
                "saves": rsim_car.saves,
                "shots": rsim_car.shots,
                "assists": rsim_car.assists,
                "demos": rsim_car.demos,
                "goals": rsim_car.goals,
                "boost_pickups": rsim_car.boost_pickups,
            }
            prev = self._prev_stats.get(agent_id, {k: 0 for k in current})
            rsim_stats[agent_id] = {k: current[k] - prev[k] for k in current}
            self._prev_stats[agent_id] = current

        return gs
