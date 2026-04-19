#pragma once
#include <RLGymCPP/Rewards/Reward.h>
#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/CommonValues.h>

namespace rl_rl {

// ── InAirReward ────────────────────────────────────────────────────────
// Port of rewards.py:InAirReward. Different from GGL's AirReward which is
// just !isOnGround — this requires the player to actually be jumping (rewards
// deliberate air, not just drives off bumps/ramps).
class InAirReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		bool jumping = (player.isJumping || player.hasDoubleJumped) && !player.isOnGround;
		return jumping ? 1.f : 0.f;
	}
};

// ── SupersonicSpeedReward ──────────────────────────────────────────────
// Port of rewards.py:SpeedReward. Uses CarState::supersonicTime — rewards
// sustained supersonic travel (capped at 3s = 1.0 reward).
class SupersonicSpeedReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		float t = player.supersonicTime;
		if (t > 3.f) t = 3.f;
		return t / 3.f;
	}
};

// ── DemoReward ─────────────────────────────────────────────────────────
// Port of rewards.py:DemoReward. Asymmetric: penalty for being demoed is
// bias-adjusted harder than reward for demoing (default bias 0.5 = symmetric).
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

// ── SpeedBumpReward ────────────────────────────────────────────────────
// Port of rewards.py:BumpReward. Zero-sum: bumper gets +speed_ratio, victim
// gets -speed_ratio, only when bump is cross-team. Must override
// GetAllRewards() to pair-up bumper/victim since the Player struct doesn't
// directly expose victim ID (we infer via carContact.otherCarID).
class SpeedBumpReward : public RLGC::Reward {
public:
	std::vector<float> GetAllRewards(const RLGC::GameState& state, bool isFinal) override {
		std::vector<float> rewards(state.players.size(), 0.f);

		for (size_t i = 0; i < state.players.size(); i++) {
			const auto& bumper = state.players[i];
			if (!bumper.eventState.bump) continue;

			uint32_t victimId = bumper.carContact.otherCarID;
			if (victimId == 0) continue;

			// Find victim in players vector by carId
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
		// Unused — GetAllRewards is overridden. Base's default calls GetReward
		// per-player but our logic is cross-player so we bypass that.
		return 0.f;
	}
};

// ── AcceleratedTouchReward ─────────────────────────────────────────────
// Port of rewards.py:AcceleratedTouchReward. On ball touch, reward is
// touch_reward + acceleration_reward * (|Δball_vel| / BALL_MAX_SPEED).
// Note: uses full velocity-vector delta (not speed delta), so the bot
// is also rewarded for redirecting the ball, not only speeding it up.
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

// ── DeltaDistToBallReward ──────────────────────────────────────────────
// Port of rewards.py:DeltaDistToBallReward. Potential-based.
// Φ(s) = -dist(car, ball). Reward = (prev_dist - dist) / (dt * CAR_MAX_SPEED).
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

// ── DeltaBallToGoalReward ──────────────────────────────────────────────
// Port of rewards.py:DeltaBallToGoalReward. Potential-based pressure on
// opponent goal. Uses BACK_WALL_Y (5120) for goal-line projection; Python
// used the same implicit value via BACK_WALL_Y constant.
class DeltaBallToGoalReward : public RLGC::Reward {
public:
	float GetReward(const RLGC::Player& player, const RLGC::GameState& state, bool isFinal) override {
		if (!state.prev) return 0.f;
		float dt = state.deltaTime;
		if (dt <= 0.f) return 0.f;

		float goalY = (player.team == RLGC::Team::BLUE)
			? RLGC::CommonValues::BACK_WALL_Y
			: -RLGC::CommonValues::BACK_WALL_Y;
		Vec goal(0.f, goalY, RLGC::CommonValues::GOAL_HEIGHT * 0.5f);

		float dist = (state.ball.pos - goal).Length();
		float prevDist = (state.prev->ball.pos - goal).Length();
		return (prevDist - dist) / (dt * RLGC::CommonValues::BALL_MAX_SPEED);
	}
};

}
