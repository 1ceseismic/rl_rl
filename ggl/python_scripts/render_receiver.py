from __future__ import annotations

import json
import traceback

import rlviser_py as rlviser
import RocketSim as rsim
from rlgym.rocket_league.common_values import BOOST_LOCATIONS

TICK_RATE = 120.0 / 8.0

rlviser.set_boost_pad_locations(BOOST_LOCATIONS)

_packet_id = 0


def get_game_speed() -> float:
    try:
        return float(rlviser.get_game_speed())
    except BaseException:  # stray KeyboardInterrupt
        return 1.0


def _vec(lst):
    return rsim.Vec(float(lst[0]), float(lst[1]), float(lst[2]))


def _rotmat(phys):
    # sequential basis, not row-major
    f = phys["forward"]; r = phys["right"]; u = phys["up"]
    return rsim.RotMat(*f, *r, *u)


def _ball_state(ball_json):
    b = rsim.BallState()
    b.pos = _vec(ball_json["pos"])
    b.vel = _vec(ball_json["vel"])
    b.ang_vel = _vec(ball_json["ang_vel"])
    b.rot_mat = _rotmat(ball_json)
    return b


def _car_state(player_json):
    phys = player_json["phys"]
    cs = rsim.CarState()
    cs.pos = _vec(phys["pos"])
    cs.vel = _vec(phys["vel"])
    cs.ang_vel = _vec(phys["ang_vel"])
    cs.rot_mat = _rotmat(phys)
    cs.boost = float(player_json.get("boost_amount", 0.0)) * 100.0
    cs.is_on_ground = bool(player_json.get("on_ground", True))
    cs.is_demoed = bool(player_json.get("is_demoed", False))
    return cs


def _gamemode(s: str):
    s = s.lower()
    if s in ("soccar", "soccer"):
        return rsim.GameMode.SOCCAR
    if s == "hoops":
        return rsim.GameMode.HOOPS
    return rsim.GameMode.SOCCAR


def render_state(state_json_str: str) -> None:
    global _packet_id
    try:
        j = json.loads(state_json_str)
        state = j.get("state", j)
        gamemode = _gamemode(j.get("gamemode", "soccar"))

        ball = _ball_state(state["ball"])

        pad_states = [bool(p) for p in state.get("boost_pads", [])]

        cars = []
        for player in state.get("players", []):
            car_id = int(player.get("car_id", len(cars) + 1))
            team = int(player.get("team_num", 0))
            cars.append((car_id, team, rsim.CarConfig(rsim.CarConfig.OCTANE), _car_state(player)))

        _packet_id += 1
        rlviser.render(
            tick_count=_packet_id,
            tick_rate=TICK_RATE,
            game_mode=gamemode,
            boost_pad_states=pad_states,
            ball=ball,
            cars=cars,
        )
    except BaseException:  # stray KeyboardInterrupt
        print("render_receiver: exception forwarding state to rlviser:")
        traceback.print_exc()
