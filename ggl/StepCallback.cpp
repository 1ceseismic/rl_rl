#include "StepCallback.h"
#include <RLGymCPP/CommonValues.h>

void rl_rl::StepCallback(GGL::Learner* learner, const std::vector<RLGC::GameState>& states, GGL::Report& report) {
	namespace CV = RLGC::CommonValues;

	for (auto& state : states) {
		for (auto& player : state.players) {
			report.AddAvg("Player/Boost", player.boost);
			report.AddAvg("Player/Speed", player.vel.Length());
			report.AddAvg("Player/Speed Pct Max", player.vel.Length() / CV::CAR_MAX_SPEED * 100.0);
			report.AddAvg("Player/Air Time Ratio", !player.isOnGround);
			report.AddAvg("Player/Supersonic Ratio", player.isSupersonic);
			report.AddAvg("Player/Demoed Ratio", player.isDemoed);
			report.AddAvg("Player/Ball Touch Ratio", player.ballTouchedStep);

			report.AddAvg("Player/Distance to Ball", (state.ball.pos - player.pos).Length());

			if (player.ballTouchedStep) {
				report.AddAvg("Game/Touch Height", state.ball.pos.z);
				report.AddAvg("Game/Touch Speed", state.ball.vel.Length());
			}

			if (player.prev && player.isFlipping && !player.prev->isFlipping) {
				report.Add("Event/Flips", 1.0);
			}
		}

		if (state.goalScored) {
			report.AddAvg("Game/Goal Speed", state.ball.vel.Length());
			if (state.ball.pos.y > 0) {
				report.Add("Game/Blue Goals", 1.0);
			} else {
				report.Add("Game/Orange Goals", 1.0);
			}
		}
	}
}
