import numpy as np
from typing import Any, Dict, List, Tuple

from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import CAR_MAX_SPEED, JUMP_MAX_TIME
from rlgym.rocket_league.obs_builders import DefaultObs

class VelocityPlayerToBallReward(RewardFunction[AgentID, GameState, float]):
    def reset(
        self,
        agents: List[AgentID],
        initial_state: GameState,
        shared_info: Dict[str, Any],
    ) -> None:
        pass

    def get_rewards(
        self,
        agents: List[AgentID],
        state: GameState,
        is_terminated: Dict[AgentID, bool],
        is_truncated: Dict[AgentID, bool],
        shared_info: Dict[str, Any],
    ) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState):
        ball = state.ball
        car = state.cars[agent].physics

        car_to_ball = ball.position - car.position
        car_to_ball = car_to_ball / np.linalg.norm(car_to_ball)

        return np.dot(car_to_ball, car.linear_velocity) / CAR_MAX_SPEED
    

class InAirReward(RewardFunction[AgentID, GameState, float]):

    def reset(self, agents: List[AgentID], inital_state: GameState, sahred_info: Dict[str, any]) -> None: pass

    def get_rewards(self, agents: List[AgentID],state: GameState) -> Dict[AgentID, float]:
        return {agent: self._get_reward(self, agents) for agent in agents}

    def get_reward(self, agent: AgentID, state: GameState):
        car = state.cars[agent]

        jumping = (car.is_jumping or car.is_holding_jump or car.has_double_jumped) and not car.on_ground

        time_in_air = car.air_time_since_jump
        return jumping + time_in_air
