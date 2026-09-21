#include "StepCallback.h"
#include <RLGymCPP/CommonValues.h>

namespace {
	struct Acc {
		double total = 0;
		uint64_t count = 0;
		void operator+=(double v) { total += v; count++; }
	};
}

void rl_rl::StepCallback(GGL::Learner* learner, const std::vector<RLGC::GameState>& states, GGL::Report& report) {
	namespace CV = RLGC::CommonValues;

	Acc boost, speed, speedPct, airTime, supersonic, demoed, touchRatio, distToBall;
	Acc touchHeight, touchSpeed, goalSpeed;
	double flips = 0, blueGoals = 0, orangeGoals = 0;

	for (auto& state : states) {
		for (auto& player : state.players) {
			double vel = player.vel.Length();
			boost += player.boost;
			speed += vel;
			speedPct += vel / CV::CAR_MAX_SPEED * 100.0;
			airTime += !player.isOnGround;
			supersonic += player.isSupersonic;
			demoed += player.isDemoed;
			touchRatio += player.ballTouchedStep;
			distToBall += (state.ball.pos - player.pos).Length();

			if (player.ballTouchedStep) {
				touchHeight += state.ball.pos.z;
				touchSpeed += state.ball.vel.Length();
			}

			if (player.prev && player.isFlipping && !player.prev->isFlipping)
				flips++;
		}

		if (state.goalScored) {
			goalSpeed += state.ball.vel.Length();
			if (state.ball.pos.y > 0) {
				blueGoals++;
			} else {
				orangeGoals++;
			}
		}
	}

	// merge once per key
	auto avg = [&report](const char* key, const Acc& acc) {
		if (!acc.count)
			return;
		auto& dst = report.avgs[key];
		dst.total += acc.total;
		dst.count += acc.count;
	};

	avg("Player/Boost", boost);
	avg("Player/Speed", speed);
	avg("Player/Speed Pct Max", speedPct);
	avg("Player/Air Time Ratio", airTime);
	avg("Player/Supersonic Ratio", supersonic);
	avg("Player/Demoed Ratio", demoed);
	avg("Player/Ball Touch Ratio", touchRatio);
	avg("Player/Distance to Ball", distToBall);
	avg("Game/Touch Height", touchHeight);
	avg("Game/Touch Speed", touchSpeed);
	avg("Game/Goal Speed", goalSpeed);

	if (flips)
		report.Add("Event/Flips", flips);
	if (blueGoals)
		report.Add("Game/Blue Goals", blueGoals);
	if (orangeGoals)
		report.Add("Game/Orange Goals", orangeGoals);
}
