#pragma once
#include <RLGymCPP/Rewards/Reward.h>
#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/CommonValues.h>

namespace rl_rl {

class InAirReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		bool jumping = (player.isJumping || player.hasDoubleJumped) && !player.isOnGround;
		return jumping ? 1.f : 0.f;
	}
};

class SupersonicSpeedReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		float t = player.supersonicTime;
		if (t > 3.f) t = 3.f;
		return t / 3.f;
	}
};

class DemoReward : public RLGC::Reward {
public:
	float aggressionBias;
	DemoReward(float aggressionBias = 0.5f) : aggressionBias(aggressionBias) {}

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		float magnitude = 0.5f + aggressionBias;
		if (player.isDemoed) return -magnitude;
		if (player.eventState.demo) return magnitude;
		return 0.f;
	}
};

class SpeedBumpReward : public RLGC::Reward {
public:
	std::vector<float> GetAllRewards(const RLGC::GameState& state, bool isFinal) override {
		std::vector<float> rewards(state.players.size(), 0.f);

		for (size_t i = 0; i < state.players.size(); i++) {
			const auto& bumper = state.players[i];
			if (!bumper.eventState.bump) continue;

			uint32_t victimId = bumper.carContact.otherCarID;
			if (victimId == 0) continue;

			int victimIdx = -1;
			for (size_t j = 0; j < state.players.size(); j++) {
				if (state.players[j].carId == victimId) { victimIdx = (int)j; break; }
			}
			if (victimIdx < 0) continue;

			const auto& victim = state.players[victimIdx];
			if (victim.team == bumper.team) continue;

			float speedRatio = bumper.vel.Length() / RLGC::CommonValues::CAR_MAX_SPEED;
			rewards[i] += speedRatio;
			rewards[victimIdx] -= speedRatio;
		}
		return rewards;
	}

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		return 0.f; // see GetAllRewards
	}
};

class AcceleratedTouchReward : public RLGC::Reward {
public:
	float touchReward;
	float accelerationReward;
	AcceleratedTouchReward(float touchReward = 0.2f, float accelerationReward = 1.f)
		: touchReward(touchReward), accelerationReward(accelerationReward) {}

	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!player.ballTouchedStep) return 0.f;
		if (!state.prev) return touchReward;
		Vec dv = state.ball.vel - state.prev->ball.vel;
		float accel = dv.Length() / RLGC::CommonValues::BALL_MAX_SPEED;
		return touchReward + accel * accelerationReward;
	}
};

class DeltaDistToBallReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!state.prev || !player.prev) return 0.f;
		float dt = state.deltaTime;
		if (dt <= 0.f) return 0.f;
		float dist = (state.ball.pos - player.pos).Length();
		float prevDist = (state.prev->ball.pos - player.prev->pos).Length();
		return (prevDist - dist) / (dt * RLGC::CommonValues::CAR_MAX_SPEED);
	}
};

class DeltaBallToGoalReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!state.prev) return 0.f;
		float dt = state.deltaTime;
		if (dt <= 0.f) return 0.f;

		float goalY = (player.team == Team::BLUE)
			? RLGC::CommonValues::BACK_WALL_Y
			: -RLGC::CommonValues::BACK_WALL_Y;
		Vec goal(0.f, goalY, RLGC::CommonValues::GOAL_HEIGHT * 0.5f);

		float dist = (state.ball.pos - goal).Length();
		float prevDist = (state.prev->ball.pos - goal).Length();
		return (prevDist - dist) / (dt * RLGC::CommonValues::BALL_MAX_SPEED);
	}
};

}
