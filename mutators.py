import random
from typing import Any, Dict, List

import numpy as np
from rlgym.api import AgentID, DoneCondition, StateMutator
from rlgym.rocket_league.api import GameState
from rlgym.rocket_league.common_values import (
    BLUE_TEAM, BALL_RESTING_HEIGHT, SIDE_WALL_X, BACK_WALL_Y, CEILING_Z,
    BALL_RADIUS, GOAL_HEIGHT, GOAL_CENTER_TO_POST, TICKS_PER_SECOND,
)

# Rocket League field reference:
#   X: -4096 to 4096 (SIDE_WALL_X)
#   Y: -5120 to 5120 (BACK_WALL_Y)
#   Z: 0 to 2044 (CEILING_Z)
#   Goal: width ~1786 (GOAL_CENTER_TO_POST*2), height 643 (GOAL_HEIGHT)
#   Corners are curved — keep ~1000uu from diagonal corners
#   Car ground Z: 17, Octane hitbox height: 46
#   Ball resting Z: 93 (BALL_RESTING_HEIGHT), Ball radius: 91 (BALL_RADIUS)
CAR_HEIGHT = 17
CAR_ROOF_Z = 46


# ── Match Mutators (normal gameplay, long episodes) ────────────────────

class AerialBallMutator(StateMutator[GameState]):
    """Ball high in the air, cars on ground with full boost."""

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        ball_x = random.uniform(-SIDE_WALL_X * 0.5, SIDE_WALL_X * 0.5)
        ball_y = random.uniform(-BACK_WALL_Y * 0.5, BACK_WALL_Y * 0.5)
        ball_z = random.uniform(400, CEILING_Z - 300)
        state.ball.position = np.array([ball_x, ball_y, ball_z], dtype=np.float32)
        state.ball.linear_velocity = np.random.uniform(-200, 200, 3).astype(np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)

        for car in state.cars.values():
            offset_x = random.uniform(-1000, 1000)
            offset_y = random.uniform(-1000, 1000)
            car_x = np.clip(ball_x + offset_x, -SIDE_WALL_X * 0.7, SIDE_WALL_X * 0.7)
            car_y = np.clip(ball_y + offset_y, -BACK_WALL_Y * 0.7, BACK_WALL_Y * 0.7)
            car.physics.position = np.array([car_x, car_y, CAR_HEIGHT], dtype=np.float32)
            angle = np.arctan2(ball_y - car_y, ball_x - car_x)
            car.physics.euler_angles = np.array([0, angle, 0], dtype=np.float32)
            car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.boost_amount = 100


class WallBallMutator(StateMutator[GameState]):
    """Ball on or near a wall. Teaches wall clears and wall play."""

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        wall = random.choice(["left", "right", "back_blue", "back_orange"])

        if wall == "left":
            ball_x = -SIDE_WALL_X + BALL_RADIUS + 50
            ball_y = random.uniform(-BACK_WALL_Y * 0.5, BACK_WALL_Y * 0.5)
        elif wall == "right":
            ball_x = SIDE_WALL_X - BALL_RADIUS - 50
            ball_y = random.uniform(-BACK_WALL_Y * 0.5, BACK_WALL_Y * 0.5)
        elif wall == "back_blue":
            ball_x = random.uniform(-SIDE_WALL_X * 0.4, SIDE_WALL_X * 0.4)
            ball_y = -BACK_WALL_Y + BALL_RADIUS + 50
        else:
            ball_x = random.uniform(-SIDE_WALL_X * 0.4, SIDE_WALL_X * 0.4)
            ball_y = BACK_WALL_Y - BALL_RADIUS - 50

        ball_z = random.uniform(300, 1000)
        state.ball.position = np.array([ball_x, ball_y, ball_z], dtype=np.float32)
        state.ball.linear_velocity = np.random.uniform(-200, 200, 3).astype(np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)

        for car in state.cars.values():
            car_x = np.clip(ball_x * 0.5 + random.uniform(-500, 500), -SIDE_WALL_X * 0.6, SIDE_WALL_X * 0.6)
            car_y = np.clip(ball_y * 0.5 + random.uniform(-500, 500), -BACK_WALL_Y * 0.6, BACK_WALL_Y * 0.6)
            car.physics.position = np.array([car_x, car_y, CAR_HEIGHT], dtype=np.float32)
            angle = np.arctan2(ball_y - car_y, ball_x - car_x)
            car.physics.euler_angles = np.array([0, angle, 0], dtype=np.float32)
            car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.boost_amount = random.uniform(50, 100)


# ── Drill Mutators (focused skill training, short episodes) ───────────

class DribbleMutator(StateMutator[GameState]):
    """Ball on car roof in own half, opponent between dribbler and goal."""

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        cars = list(state.cars.values())
        if not cars:
            return

        dribbler = random.choice(cars)
        is_blue = dribbler.team_num == BLUE_TEAM

        # Dribbler in own half
        car_x = random.uniform(-SIDE_WALL_X * 0.3, SIDE_WALL_X * 0.3)
        car_y = random.uniform(-BACK_WALL_Y * 0.1, -BACK_WALL_Y * 0.5) if is_blue else \
                random.uniform(BACK_WALL_Y * 0.1, BACK_WALL_Y * 0.5)

        goal_y = BACK_WALL_Y if is_blue else -BACK_WALL_Y
        angle = np.arctan2(goal_y - car_y, -car_x * 0.1)

        dribbler.physics.position = np.array([car_x, car_y, CAR_HEIGHT], dtype=np.float32)
        dribbler.physics.euler_angles = np.array([0, angle, 0], dtype=np.float32)
        speed = random.uniform(800, 1400)
        dribbler.physics.linear_velocity = np.array(
            [np.cos(angle) * speed, np.sin(angle) * speed, 0], dtype=np.float32
        )
        dribbler.physics.angular_velocity = np.zeros(3, dtype=np.float32)
        dribbler.boost_amount = random.uniform(30, 70)

        ball_z = CAR_HEIGHT + CAR_ROOF_Z + BALL_RADIUS
        state.ball.position = np.array([car_x, car_y, ball_z], dtype=np.float32)
        state.ball.linear_velocity = dribbler.physics.linear_velocity.copy()
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)

        for car in cars:
            if car is dribbler:
                continue
            opp_x = np.clip(car_x + random.uniform(-800, 800), -SIDE_WALL_X * 0.5, SIDE_WALL_X * 0.5)
            if is_blue:
                opp_y = min(car_y + random.uniform(1500, 3000), BACK_WALL_Y * 0.8)
            else:
                opp_y = max(car_y - random.uniform(1500, 3000), -BACK_WALL_Y * 0.8)

            car.physics.position = np.array([opp_x, opp_y, CAR_HEIGHT], dtype=np.float32)
            opp_angle = np.arctan2(car_y - opp_y, car_x - opp_x)
            car.physics.euler_angles = np.array([0, opp_angle, 0], dtype=np.float32)
            car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.boost_amount = random.uniform(50, 100)


class RecoveryMutator(StateMutator[GameState]):
    """Car spawned airborne at a random orientation. Must land and reach the ball."""

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        # Ball on ground somewhere
        ball_x = random.uniform(-SIDE_WALL_X * 0.5, SIDE_WALL_X * 0.5)
        ball_y = random.uniform(-BACK_WALL_Y * 0.4, BACK_WALL_Y * 0.4)
        state.ball.position = np.array([ball_x, ball_y, BALL_RESTING_HEIGHT], dtype=np.float32)
        state.ball.linear_velocity = np.random.uniform(-300, 300, 3).astype(np.float32)
        state.ball.linear_velocity[2] = abs(state.ball.linear_velocity[2])  # slight upward
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)

        for car in state.cars.values():
            # Airborne near the ball
            car_x = ball_x + random.uniform(-1500, 1500)
            car_y = ball_y + random.uniform(-1500, 1500)
            car_x = np.clip(car_x, -SIDE_WALL_X * 0.7, SIDE_WALL_X * 0.7)
            car_y = np.clip(car_y, -BACK_WALL_Y * 0.6, BACK_WALL_Y * 0.6)
            car_z = random.uniform(300, 800)

            car.physics.position = np.array([car_x, car_y, car_z], dtype=np.float32)
            # Random orientation — tumbling
            pitch = random.uniform(-np.pi, np.pi)
            yaw = random.uniform(-np.pi, np.pi)
            roll = random.uniform(-np.pi, np.pi)
            car.physics.euler_angles = np.array([pitch, yaw, roll], dtype=np.float32)
            # Some momentum
            car.physics.linear_velocity = np.random.uniform(-500, 500, 3).astype(np.float32)
            car.physics.angular_velocity = np.random.uniform(-3, 3, 3).astype(np.float32)
            car.boost_amount = random.uniform(20, 60)


class ShotSetupMutator(StateMutator[GameState]):
    """Ball rolling toward car from an angle, with a clear shot at goal."""

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        cars = list(state.cars.values())
        if not cars:
            return

        shooter = random.choice(cars)
        is_blue = shooter.team_num == BLUE_TEAM

        # Shooter in opponent half, facing goal
        goal_y = BACK_WALL_Y if is_blue else -BACK_WALL_Y
        car_x = random.uniform(-SIDE_WALL_X * 0.4, SIDE_WALL_X * 0.4)
        car_y = (BACK_WALL_Y * random.uniform(0.1, 0.4)) if is_blue else \
                (-BACK_WALL_Y * random.uniform(0.1, 0.4))

        angle = np.arctan2(goal_y - car_y, -car_x * 0.3)
        shooter.physics.position = np.array([car_x, car_y, CAR_HEIGHT], dtype=np.float32)
        shooter.physics.euler_angles = np.array([0, angle, 0], dtype=np.float32)
        speed = random.uniform(500, 1200)
        shooter.physics.linear_velocity = np.array(
            [np.cos(angle) * speed, np.sin(angle) * speed, 0], dtype=np.float32
        )
        shooter.physics.angular_velocity = np.zeros(3, dtype=np.float32)
        shooter.boost_amount = random.uniform(30, 80)

        # Ball rolling toward the shooter from an angle (a "pass")
        ball_offset_x = random.uniform(-1000, 1000)
        ball_ahead = random.uniform(500, 1500)
        if is_blue:
            ball_y = car_y + ball_ahead
        else:
            ball_y = car_y - ball_ahead
        ball_x = car_x + ball_offset_x
        ball_x = np.clip(ball_x, -SIDE_WALL_X * 0.6, SIDE_WALL_X * 0.6)
        ball_y = np.clip(ball_y, -BACK_WALL_Y * 0.8, BACK_WALL_Y * 0.8)

        state.ball.position = np.array([ball_x, ball_y, BALL_RESTING_HEIGHT], dtype=np.float32)
        # Ball rolling toward shooter
        ball_vel_x = (car_x - ball_x) * random.uniform(0.3, 0.8)
        ball_vel_y = (car_y - ball_y) * random.uniform(0.3, 0.8)
        state.ball.linear_velocity = np.array([ball_vel_x, ball_vel_y, 0], dtype=np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)

        # Opponent in goal area
        for car in cars:
            if car is shooter:
                continue
            opp_x = random.uniform(-GOAL_CENTER_TO_POST * 0.8, GOAL_CENTER_TO_POST * 0.8)
            if is_blue:
                opp_y = BACK_WALL_Y * random.uniform(0.7, 0.9)
            else:
                opp_y = -BACK_WALL_Y * random.uniform(0.7, 0.9)
            car.physics.position = np.array([opp_x, opp_y, CAR_HEIGHT], dtype=np.float32)
            opp_angle = np.arctan2(ball_y - opp_y, ball_x - opp_x)
            car.physics.euler_angles = np.array([0, opp_angle, 0], dtype=np.float32)
            car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.boost_amount = random.uniform(30, 60)


class DefenseMutator(StateMutator[GameState]):
    """Ball heading toward own goal. Must save/clear."""

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        cars = list(state.cars.values())
        if not cars:
            return

        defender = random.choice(cars)
        is_blue = defender.team_num == BLUE_TEAM

        # Defender near own goal
        own_goal_y = -BACK_WALL_Y if is_blue else BACK_WALL_Y
        car_x = random.uniform(-GOAL_CENTER_TO_POST * 0.8, GOAL_CENTER_TO_POST * 0.8)
        car_y = own_goal_y * random.uniform(0.7, 0.9)

        defender.physics.position = np.array([car_x, car_y, CAR_HEIGHT], dtype=np.float32)
        angle = np.arctan2(-own_goal_y - car_y, -car_x)  # face away from own goal
        defender.physics.euler_angles = np.array([0, angle, 0], dtype=np.float32)
        defender.physics.linear_velocity = np.zeros(3, dtype=np.float32)
        defender.physics.angular_velocity = np.zeros(3, dtype=np.float32)
        defender.boost_amount = random.uniform(20, 50)

        # Ball coming toward own goal from midfield
        ball_x = random.uniform(-SIDE_WALL_X * 0.4, SIDE_WALL_X * 0.4)
        ball_y_sign = -1 if is_blue else 1
        ball_y = ball_y_sign * BACK_WALL_Y * random.uniform(0.1, 0.4)
        ball_z = random.uniform(BALL_RESTING_HEIGHT, 600)

        state.ball.position = np.array([ball_x, ball_y, ball_z], dtype=np.float32)
        # Ball velocity aimed at goal
        goal_x = random.uniform(-GOAL_CENTER_TO_POST * 0.6, GOAL_CENTER_TO_POST * 0.6)
        ball_speed = random.uniform(1500, 3000)
        to_goal = np.array([goal_x - ball_x, own_goal_y - ball_y, 0], dtype=np.float32)
        to_goal = to_goal / (np.linalg.norm(to_goal) + 1e-6)
        state.ball.linear_velocity = (to_goal * ball_speed).astype(np.float32)
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)

        # Attacker chasing from behind the ball
        for car in cars:
            if car is defender:
                continue
            att_x = ball_x + random.uniform(-500, 500)
            if is_blue:
                att_y = ball_y + random.uniform(500, 1500)  # behind ball (further from blue goal)
            else:
                att_y = ball_y - random.uniform(500, 1500)
            att_x = np.clip(att_x, -SIDE_WALL_X * 0.6, SIDE_WALL_X * 0.6)
            att_y = np.clip(att_y, -BACK_WALL_Y * 0.8, BACK_WALL_Y * 0.8)
            car.physics.position = np.array([att_x, att_y, CAR_HEIGHT], dtype=np.float32)
            att_angle = np.arctan2(own_goal_y - att_y, goal_x - att_x)
            car.physics.euler_angles = np.array([0, att_angle, 0], dtype=np.float32)
            att_speed = random.uniform(1000, 2000)
            car.physics.linear_velocity = np.array(
                [np.cos(att_angle) * att_speed, np.sin(att_angle) * att_speed, 0], dtype=np.float32
            )
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.boost_amount = random.uniform(40, 80)


class FastAerialMutator(StateMutator[GameState]):
    """Ball launched upward from nearby. Car on ground, must fast-aerial to reach it."""

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        # Ball just above the car, moving upward
        ball_x = random.uniform(-SIDE_WALL_X * 0.4, SIDE_WALL_X * 0.4)
        ball_y = random.uniform(-BACK_WALL_Y * 0.4, BACK_WALL_Y * 0.4)
        ball_z = random.uniform(250, 500)
        ball_vz = random.uniform(500, 1200)  # upward velocity

        state.ball.position = np.array([ball_x, ball_y, ball_z], dtype=np.float32)
        state.ball.linear_velocity = np.array(
            [random.uniform(-200, 200), random.uniform(-200, 200), ball_vz], dtype=np.float32
        )
        state.ball.angular_velocity = np.zeros(3, dtype=np.float32)

        for car in state.cars.values():
            car_x = ball_x + random.uniform(-500, 500)
            car_y = ball_y + random.uniform(-500, 500)
            car_x = np.clip(car_x, -SIDE_WALL_X * 0.6, SIDE_WALL_X * 0.6)
            car_y = np.clip(car_y, -BACK_WALL_Y * 0.6, BACK_WALL_Y * 0.6)
            car.physics.position = np.array([car_x, car_y, CAR_HEIGHT], dtype=np.float32)
            angle = np.arctan2(ball_y - car_y, ball_x - car_x)
            car.physics.euler_angles = np.array([0, angle, 0], dtype=np.float32)
            car.physics.linear_velocity = np.zeros(3, dtype=np.float32)
            car.physics.angular_velocity = np.zeros(3, dtype=np.float32)
            car.boost_amount = 100


# ── Randomizer + Scenario-aware timeout ───────────────────────────────

# Drill mutators get short episodes; match mutators get full length
DRILL_MUTATORS = (DribbleMutator, RecoveryMutator, ShotSetupMutator, DefenseMutator, FastAerialMutator)

class RandomMutator(StateMutator[GameState]):
    """Randomly picks from a weighted set of mutators. Tags drills in shared_info."""

    def __init__(self, *weighted_mutators: tuple):
        """Each argument is (mutator, weight). Example: (KickoffMutator(), 0.5)"""
        self.mutators = [m for m, w in weighted_mutators]
        weights = [w for m, w in weighted_mutators]
        total = sum(weights)
        self.weights = [w / total for w in weights]

    def apply(self, state: GameState, shared_info: Dict[str, Any]) -> None:
        chosen = random.choices(self.mutators, weights=self.weights, k=1)[0]
        chosen.apply(state, shared_info)
        shared_info["is_drill"] = isinstance(chosen, DRILL_MUTATORS)


class DrillTimeoutCondition(DoneCondition[AgentID, GameState]):
    """Short timeout for drills, long timeout for match episodes."""

    def __init__(self, match_timeout_seconds: float = 300, drill_timeout_seconds: float = 15):
        self.match_timeout = match_timeout_seconds
        self.drill_timeout = drill_timeout_seconds
        self.initial_tick = None
        self.is_drill = False

    def reset(self, agents: List[AgentID], initial_state: GameState, shared_info: Dict[str, Any]) -> None:
        self.initial_tick = initial_state.tick_count
        self.is_drill = shared_info.get("is_drill", False)

    def is_done(self, agents: List[AgentID], state: GameState, shared_info: Dict[str, Any]) -> Dict[AgentID, bool]:
        timeout = self.drill_timeout if self.is_drill else self.match_timeout
        elapsed = (state.tick_count - self.initial_tick) / TICKS_PER_SECOND
        done = elapsed >= timeout
        return {agent: done for agent in agents}
