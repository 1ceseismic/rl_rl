#pragma once
#include <GigaLearnCPP/Learner.h>

namespace rl_rl {

// Port of metrics.py:CustomMetricsProvider. Populates Report with:
//   Player/Avg Boost, Player/Avg Speed, Player/Air Time Ratio,
//   Player/Supersonic Ratio, Player/Ball Touch Ratio, Player/Touch Height,
//   Game/Goal Speed, Game/Blue Goals, Game/Orange Goals
//
// Running averages are managed by Report::AddAvg() which uses a Welford
// accumulator internally — matches Python's running-mean math.
//
// Event counters (demos, saves, shots, assists, bumps) are already added
// automatically by GGL when addRewardsToMetrics=true and built-in event
// rewards are wired in. We still emit touch/speed/boost here since those
// aren't per-reward scalars.
void StepCallback(GGL::Learner* learner, const std::vector<RLGC::GameState>& states, GGL::Report& report);

}
