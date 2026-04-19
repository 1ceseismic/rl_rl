#pragma once
#include <RLGymCPP/ObsBuilders/DefaultObs.h>

namespace rl_rl {

// DefaultObs with exactly the normalization coefficients from train.py.
// Uses BACK_NET_Y (6000) instead of DefaultObs's BACK_WALL_Y (5120) for Y
// so ball positions behind the goal (backboard plays) still normalize to [-1, 1].
//
// Note: GGL's DefaultObs encodes car orientation via rotation-matrix forward/up
// vectors (unit-length, no pi-normalization), while Python rlgym's DefaultObs
// uses euler angles with ang_coef=1/pi. This is an encoding difference; the
// obs size differs from Python by 3 scalars per car (6 rotation vs 3 euler).
// Checkpoints from Python are NOT directly loadable into GGL for this reason.
class NormalizedObs : public RLGC::DefaultObs {
public:
	static constexpr float BACK_NET_Y = 6000.f;

	NormalizedObs() : RLGC::DefaultObs(
		Vec(
			1.f / RLGC::CommonValues::SIDE_WALL_X,
			1.f / BACK_NET_Y,
			1.f / RLGC::CommonValues::CEILING_Z
		),
		1.f / RLGC::CommonValues::CAR_MAX_SPEED,
		1.f / RLGC::CommonValues::CAR_MAX_ANG_VEL
	) {}
};

}
