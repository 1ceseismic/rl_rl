import numpy as np
from typing import Any, Dict, List, Tuple

from rlgym.api import AgentID, RewardFunction
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import (
    CAR_MAX_SPEED, BALL_MAX_SPEED, JUMP_MAX_TIME, CEILING_Z,
    BACK_WALL_Y, GOAL_HEIGHT, BLUE_TEAM, TICKS_PER_SECOND,
)
from rlgym.rocket_league.obs_builders import DefaultObs
from rlgym.rocket_league.reward_functions import CombinedReward
from engine import rsim_stats

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
    

class DeltaDistToBallReward(RewardFunction):
    """Potential-based: rewards rate of closing distance to ball.
    Unlike VelocityPlayerToBallReward, properly accounts for ball movement.
    Φ(s) = -dist(car, ball). Reward = Φ(s') - Φ(s), normalized by dt * max_speed.
    """

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self._prev_tick = initial_state.tick_count
        self._prev_dist = {
            agent: np.linalg.norm(initial_state.ball.position - initial_state.cars[agent].physics.position)
            for agent in agents
        }

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        tick = state.tick_count
        dt = (tick - self._prev_tick) / TICKS_PER_SECOND
        self._prev_tick = tick

        rewards = {}
        for agent in agents:
            dist = np.linalg.norm(state.ball.position - state.cars[agent].physics.position)
            prev = self._prev_dist.get(agent, dist)
            rewards[agent] = (prev - dist) / (dt * CAR_MAX_SPEED) if dt > 0 else 0.0
            self._prev_dist[agent] = dist
        return rewards


class DeltaBallToGoalReward(RewardFunction):
    """Potential-based: rewards ball moving toward opponent goal.
    Φ(s) = -dist(ball, opp_goal). Reward = Φ(s') - Φ(s), normalized by dt * max_ball_speed.
    Replaces absolute speed rewards with a meaningful offensive pressure signal.
    """

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self._prev_tick = initial_state.tick_count
        self._prev_dist = {}
        for agent in agents:
            goal = self._opp_goal(initial_state.cars[agent].team_num)
            self._prev_dist[agent] = np.linalg.norm(initial_state.ball.position - goal)

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        tick = state.tick_count
        dt = (tick - self._prev_tick) / TICKS_PER_SECOND
        self._prev_tick = tick

        rewards = {}
        for agent in agents:
            goal = self._opp_goal(state.cars[agent].team_num)
            dist = np.linalg.norm(state.ball.position - goal)
            prev = self._prev_dist.get(agent, dist)
            rewards[agent] = (prev - dist) / (dt * BALL_MAX_SPEED) if dt > 0 else 0.0
            self._prev_dist[agent] = dist
        return rewards

    @staticmethod
    def _opp_goal(team_num: int) -> np.ndarray:
        goal_y = BACK_WALL_Y if team_num == BLUE_TEAM else -BACK_WALL_Y
        return np.array([0, goal_y, GOAL_HEIGHT / 2], dtype=np.float32)


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

    POSSESSION_DIST = 250  # ~2 car lengths — closer than this = dribbling, no penalty
    SLOW_THRESHOLD = 0.5   # below 50% max speed = penalized (when not in possession)

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        car = state.cars[agent]
        speed = np.linalg.norm(car.physics.linear_velocity)
        speed_ratio = speed / CAR_MAX_SPEED

        to_ball = state.ball.position - car.physics.position
        dist = np.linalg.norm(to_ball)

        # Slow penalty: going below threshold speed while far from ball
        if dist > self.POSSESSION_DIST and speed_ratio < self.SLOW_THRESHOLD:
            return speed_ratio - self.SLOW_THRESHOLD  # negative, range [~-0.5, 0)

        if speed < 1:
            return 0.0

        # Only reward speed when facing toward the ball (within ~90 degree cone)
        if dist > 0:
            to_ball /= dist
        alignment = np.dot(car.physics.forward, to_ball)  # 1=facing ball, -1=away
        if alignment < 0:
            return 0.0

        return alignment * speed_ratio


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
                if opp_dist < 400 and ball_dist > 2000:
                    return max(-400/opp_dist, -1)  # flat penalty
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


class AerialHitReward(RewardFunction):
    """Rewards touching the ball at height. Scales with how high the contact is."""

    GROUND_THRESHOLD = 300  # below this is ground play (ball radius + single jump height)

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        car = state.cars[agent]
        if car.ball_touches == 0:
            return 0.0

        height = max(state.ball.position[2], car.physics.position[2])
        normalized = max(0.0, (height - self.GROUND_THRESHOLD)) / (CEILING_Z - self.GROUND_THRESHOLD)
        return min(normalized, 1.0)


class WallJumpReward(RewardFunction):
    """Rewards jumping or flipping off walls. Detects wall by being on a surface at height."""

    WALL_HEIGHT_THRESHOLD = 200  # must be above this z to count as "on wall"

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self._was_on_wall = {}

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        rewards = {}
        for agent in agents:
            car = state.cars[agent]
            on_wall = car.on_ground and car.physics.position[2] > self.WALL_HEIGHT_THRESHOLD
            was_on_wall = self._was_on_wall.get(agent, False)

            # Reward: was on wall last tick, now jumping/flipping off it
            if was_on_wall and not car.on_ground and (car.has_jumped or car.is_flipping):
                rewards[agent] = 1.0
            else:
                rewards[agent] = 0.0

            self._was_on_wall[agent] = on_wall
        return rewards


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



class ShotReward(RewardFunction):
    """Rewards shots on goal using RocketSim's built-in shot detection."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: float(rsim_stats.get(agent, {}).get("shots", 0)) for agent in agents}


class SaveReward(RewardFunction):
    """Rewards saves using RocketSim's built-in save detection."""

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: float(rsim_stats.get(agent, {}).get("saves", 0)) for agent in agents}


class DemoReward(RewardFunction):
    def __init__(self, aggression_bias=0.5):
        self.aggression_bias = aggression_bias

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        pass

    def get_rewards(self, agents: List[AgentID], state: GameState, is_terminated: Dict[AgentID, bool],
                    is_truncated: Dict[AgentID, bool], shared_info: Dict[str, Any]) -> Dict[AgentID, float]:
        return {agent: self._get_reward(agent, state) for agent in agents}

    def _get_reward(self, agent: AgentID, state: GameState) -> float:
        car = state.cars[agent]

        if car.is_demoed:
            return -(0.5 + self.aggression_bias)  # punish dying harder than reward for killing

        if car.bump_victim_id is not None and state.cars[car.bump_victim_id].is_demoed:
            return 0.5 + self.aggression_bias  # demo'd someone and survived

        return 0.0


class TrackedCombinedReward(CombinedReward):
    """CombinedReward that stores per-component reward averages in shared_info."""

    def reset(self, agents, initial_state, shared_info):
        super().reset(agents, initial_state, shared_info)
        shared_info["reward_components"] = {
            type(fn).__name__: 0.0 for fn in self.reward_fns
        }
        shared_info["_reward_step_count"] = 0

    def get_rewards(self, agents, state, is_terminated, is_truncated, shared_info):
        shared_info["_reward_step_count"] = shared_info.get("_reward_step_count", 0) + 1
        n = shared_info["_reward_step_count"]
        components = shared_info.get("reward_components", {})

        combined_rewards = {agent: 0.0 for agent in agents}
        for reward_fn, weight in zip(self.reward_fns, self.weights):
            rewards = reward_fn.get_rewards(agents, state, is_terminated, is_truncated, shared_info)
            name = type(reward_fn).__name__

            # Average the raw (unweighted) reward across agents this tick
            avg_raw = sum(rewards.values()) / len(rewards) if rewards else 0.0
            # Running average across ticks
            components[name] = components.get(name, 0.0) + (avg_raw - components.get(name, 0.0)) / n

            for agent, reward in rewards.items():
                combined_rewards[agent] += reward * weight

        shared_info["reward_components"] = components
        return combined_rewards




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