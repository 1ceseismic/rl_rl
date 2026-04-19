#pragma once
#include <RLGymCPP/StateSetters/StateSetter.h>
#include <RLGymCPP/Math.h>
#include <RLGymCPP/CommonValues.h>

namespace rl_rl {

// Port of mutators.py:RandomFieldMutator. Aggressive random state:
// cars can be airborne, ball at full speed. This is the 40% branch of
// the training mutator mix (the other 60% is KickoffState).
class RandomFieldState : public RLGC::StateSetter {
public:
	void ResetArena(Arena* arena) override {
		using RocketSim::Math::RandFloat;
		using RLGC::Math::RandVec;
		namespace CV = RLGC::CommonValues;

		constexpr float CAR_HEIGHT = 17.f;

		// Reset boost pads etc.
		arena->ResetToRandomKickoff();

		// Ball: anywhere on field, 2000uu from goal lines, any height up to 70% ceiling
		BallState bs = {};
		bs.pos = Vec(
			RandFloat(-CV::SIDE_WALL_X * 0.7f, CV::SIDE_WALL_X * 0.7f),
			RandFloat(-CV::BACK_WALL_Y * 0.6f, CV::BACK_WALL_Y * 0.6f),
			RandFloat(CV::BALL_RADIUS, CV::CEILING_Z * 0.7f)
		);
		bs.vel = RandVec(Vec(-2300, -2300, -2300), Vec(2300, 2300, 2300));
		bs.angVel = RandVec(Vec(-3, -3, -3), Vec(3, 3, 3));
		arena->ball->SetState(bs);

		// Cars: can be airborne, random orientation, random velocity and spin
		for (Car* car : arena->_cars) {
			CarState cs = {};
			cs.pos = Vec(
				RandFloat(-CV::SIDE_WALL_X * 0.7f, CV::SIDE_WALL_X * 0.7f),
				RandFloat(-CV::BACK_WALL_Y * 0.7f, CV::BACK_WALL_Y * 0.7f),
				RandFloat(CAR_HEIGHT, CV::CEILING_Z / 6.f)
			);

			Angle angle(
				RandFloat(-M_PI, M_PI),           // yaw
				RandFloat(-M_PI * 0.5f, M_PI * 0.5f), // pitch
				RandFloat(-M_PI, M_PI)            // roll
			);
			cs.rotMat = angle.ToRotMat();

			cs.vel = RandVec(Vec(-2300, -2300, -2300), Vec(2300, 2300, 2300));
			cs.angVel = RandVec(Vec(-5.5f, -5.5f, -5.5f), Vec(5.5f, 5.5f, 5.5f));
			cs.boost = RandFloat(0, 100);

			car->SetState(cs);
		}
	}
};

}
