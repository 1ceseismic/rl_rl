"""rlviser-backed render receiver for GGL.

GGL's default render_receiver.py ships JSON over UDP to RocketSimVis on port
9273. The rest of this project uses rlviser (Bevy-based, different binary
protocol on a different port), so that receiver hits nothing and you get no
window. This replacement parses GGL's JSON payload and forwards it via
rlviser_py, which handles its own protocol + auto-launches the rlviser binary.

Copied over GGL's default at build time — see scripts/build_ggl.sh.
"""
from __future__ import annotations

import json
import traceback

import rlviser_py as rlviser
import RocketSim as rsim
from rlgym.rocket_league.common_values import BOOST_LOCATIONS

TICK_RATE = 120.0 / 8.0  # matches train.py:RLViserRenderer(tick_rate=120/8)

rlviser.set_boost_pad_locations(BOOST_LOCATIONS)

_packet_id = 0


def get_game_speed() -> float:
    """Polled by GGL's RenderSender each frame. Returns the rlviser UI
    slider value so moving it actually speeds up / slows down the sim."""
    try:
        return float(rlviser.get_game_speed())
    except BaseException:
        # BaseException (not Exception) because Python's embedded SIGINT
        # handler raises KeyboardInterrupt on stray Ctrl-C in the terminal
        # — letting that propagate to C++ crashes the training loop.
        return 1.0


def _vec(lst):
    return rsim.Vec(float(lst[0]), float(lst[1]), float(lst[2]))


def _rotmat(phys):
    # rsim.RotMat's 9-float constructor takes the basis vectors sequentially:
    #   RotMat(fx, fy, fz,  rx, ry, rz,  ux, uy, uz)
    # (verified empirically with distinct values). Passing row-major indices
    # here shuffles every car's orientation and rlviser displays cars sliding
    # across the field without rotating.
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

        # rlviser_py wants boost_pad_states as Sequence[bool]: True = available.
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
    except BaseException:
        # BaseException catches KeyboardInterrupt too. Python's embedded
        # SIGINT handler raises it on any stray Ctrl-C (even accidental
        # ones from terminal focus changes); bubbling that up to GGL's
        # RenderSender::Send() turns it into RG_ERR_CLOSE → SIGABRT and
        # kills the whole training run.
        print("render_receiver: exception forwarding state to rlviser:")
        traceback.print_exc()
