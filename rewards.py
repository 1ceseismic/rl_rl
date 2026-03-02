import numpy as np
from typing import Any, Dict, List, Tuple

from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import CAR_MAX_SPEED, BALL_MAX_SPEED, JUMP_MAX_TIME
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
        car = state.cars[agent]

        jumping = (car.is_jumping or car.has_double_jumped) and not car.on_ground
        #return float(jumping) + min(car.air_time_since_jump/5.0, 1.0)
        return float(jumping)


class FaceForwardReward(RewardFunction[AgentID, GameState, float]):
    """Rewards hitting ball with nose, penalizes hitting with back of car."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        car = state.cars[agent]
        if car.ball_touches > 0:
            to_ball = state.ball.position - car.physics.position
            dist = np.linalg.norm(to_ball)
            if dist > 0:
                to_ball /= dist
            alignment = np.dot(car.physics.forward, to_ball)  # 1=nose, -1=back
            return float(alignment)
        return 0.0



class SpeedReward(RewardFunction[AgentID, GameState, float]):
    """Rewards being supersonic."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        car = state.cars[agent]
        return min(car.supersonic_time, 3.0) / 3.0


class DenseSpeedReward(RewardFunction[AgentID, GameState, float]):
    """Rewards any speed, not just supersonic. Fires every tick."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        speed = np.linalg.norm(state.cars[agent].physics.linear_velocity)
        return speed / CAR_MAX_SPEED  # 0 = still, 1 = max speed


class BoostManagementReward(RewardFunction[AgentID, GameState, float]):
    """Penalizes hoarding boost while slow, rewards using boost to go fast."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        car = state.cars[agent]
        boost = car.boost_amount / 100.0  # [0, 1]
        speed = np.linalg.norm(car.physics.linear_velocity) / CAR_MAX_SPEED  # [0, 1]
        # High boost + low speed = hoarding = negative
        # Low boost + high speed = using it well = positive
        return speed - boost  # range [-1, 1]


class TouchBallVelocityReward(RewardFunction[AgentID, GameState, float]):
    """On ball touch, rewards based on how fast the ball is moving. Encourages hard hits (flips) over gentle nudges."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        if state.cars[agent].ball_touches > 0:
            ball_speed = np.linalg.norm(state.ball.linear_velocity)
            return ball_speed / BALL_MAX_SPEED  # 0 = dead ball, 1 = max speed
        return 0.0


class OpponentProximityPenalty(RewardFunction[AgentID, GameState, float]):
    """Penalizes being close to opponent while far from ball. Breaks rule 1 / car hugging."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        car = state.cars[agent]
        my_pos = car.physics.position
        ball_dist = np.linalg.norm(state.ball.position - my_pos)

        # Find opponent
        for other_id, other_car in state.cars.items():
            if other_car.team_num != car.team_num:
                opp_dist = np.linalg.norm(other_car.physics.position - my_pos)
                # Close to opponent (<500uu ≈ ~2.5 car lengths) and far from ball (>1000uu)
                if opp_dist < 500 and ball_dist > 1000:
                    return -1.0  # flat penalty
        return 0.0


class FlipReward(RewardFunction[AgentID, GameState, float]):
    """Small reward for flipping at all — encourages the bot to explore the mechanic."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        return float(state.cars[agent].is_flipping)


class FlipHitReward(RewardFunction[AgentID, GameState, float]):
    """Big reward for touching ball while flipping. Directly incentivizes flip shots."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        car = state.cars[agent]
        if car.ball_touches > 0 and car.is_flipping:
            ball_speed = np.linalg.norm(state.ball.linear_velocity)
            return ball_speed / BALL_MAX_SPEED  # harder flip hit = more reward
        return 0.0


class GoalRatioReward(RewardFunction):
    def __init__(self, bias=0.25):
        self.bias = bias

    def reset(self, agents, initial_state, shared_info):
        pass

    def get_rewards(self, agents, state: GameState, is_terminated, is_truncated, shared_info):
        rewards = {}
        for agent in agents:
            aggression_bias = self.bias #X % more aggressive
            goal_reward = 10
            concede_reward = -goal_reward * (1 - aggression_bias)

            if state.goal_scored:
                agent_team = state.cars[agent].team_num  # 0=Blue, 1=Orange
                if agent_team == state.scoring_team:
                    rewards[agent] = goal_reward
                else:
                    rewards[agent] = concede_reward
            else:
                rewards[agent] = 0.0
        return rewards









# '''
# This is a wrapper to put around an existing reward
# Usage: ZeroSumReward(YourOtherReward(), team_spirit, [optional: opp_scale])
# NOTE: Due to limitations in rlgym-sim, "previous_action" is not supported and will be "None" for the child reward
# '''
# class ZeroSumReward(RewardFunction):
#     '''
#     child_reward: The underlying reward function
#     team_spirit: How much to share this reward with teammates (0-1)
#     opp_scale: How to scale the penalty we get for the opponents getting this reward (usually 1)
#     '''
#     def __init__(self, child_reward: RewardFunction, team_spirit, opp_scale = 1.0):
#         self.child_reward = child_reward # type: RewardFunction
#         self.team_spirit = team_spirit
#         self.opp_scale = opp_scale

#         self._update_next = True
#         self._rewards_cache = {}

#     def reset(self, initial_state: GameState):
#         self.child_reward.reset(initial_state)

#     def pre_step(self, state: GameState):
#         self.child_reward.pre_step(state)

#         # Mark the next get_reward call as being the first reward call of the step
#         self._update_next = True

#     def update(self, state: GameState, is_final):
#         self._rewards_cache = {}

#         '''
#         Each player's reward is calculated using this equation:
#         reward = individual_rewards * (1-team_spirit) + avg_team_reward * team_spirit - avg_opp_reward * opp_scale
#         '''

#         # Get the individual rewards from each player while also adding them to that team's reward list
#         individual_rewards = {}
#         team_reward_lists = [ [], [] ]
#         for player in state.players:
#             if is_final:
#                 reward = self.child_reward.get_final_reward(player, state, None)
#             else:
#                 reward = self.child_reward.get_reward(player, state, None)
#             individual_rewards[player.car_id] = reward
#             team_reward_lists[int(player.team_num)].append(reward)

#         # If a team has no players, add a single 0 to their team rewards so the average doesn't break
#         for i in range(2):
#             if len(team_reward_lists[i]) == 0:
#                 team_reward_lists[i].append(0)

#         # Turn the team-sorted reward lists into averages for each time
#         # Example:
#         #    Before: team_rewards = [ [1, 3], [4, 8] ]
#         #    After:  team_rewards = [ 2, 6 ]
#         team_rewards = np.average(team_reward_lists, 1)

#         # Compute and cache:
#         # reward = individual_rewards * (1-team_spirit)
#         #          + avg_team_reward * team_spirit
#         #          - avg_opp_reward * opp_scale
#         for player in state.players:
#             self._rewards_cache[player.car_id] = (
#                     individual_rewards[player.car_id] * (1 - self.team_spirit)
#                     + team_rewards[int(player.team_num)] * self.team_spirit
#                     - team_rewards[1 - int(player.team_num)] * self.opp_scale
#             )

#     '''
#     I made get_reward and get_final_reward both call get_reward_multi, using the "is_final" argument to distinguish
#     Otherwise I would have to rewrite this function for final rewards, which is lame
#     '''
#     def get_reward_multi(self, player: PlayerData, state: GameState, previous_action: np.ndarray, is_final) -> float:
#         # If this is the first get_reward call this step, we need to update the rewards for all players
#         if self._update_next:
#             self.update(state, is_final)
#             self._update_next = False
#         return self._rewards_cache[player.car_id]

#     def get_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
#         return self.get_reward_multi(player, state, previous_action, False)

#     def get_final_reward(self, player: PlayerData, state: GameState, previous_action: np.ndarray) -> float:
#         return self.get_reward_multi(player, state, previous_action, True)