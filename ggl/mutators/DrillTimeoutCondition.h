#pragma once
#include <RLGymCPP/TerminalConditions/TerminalCondition.h>

namespace rl_rl {

// Port of mutators.py:DrillTimeoutCondition. Python version reads
// shared_info["is_drill"] to pick between drill/match timeout; GGL has no
// shared-info concept, so this C++ version is match-timeout only.
//
// The training config currently uses 60% kickoff / 40% random (neither is a
// drill), so only the match timeout is actually exercised. If drills are
// reintroduced later, extend this by exposing a setter on the composite
// state setter that flips an internal flag.
class DrillTimeoutCondition : public RLGC::TerminalCondition {
public:
	float matchTimeout;
	float elapsed = 0.f;

	DrillTimeoutCondition(float matchTimeoutSeconds = 300.f) : matchTimeout(matchTimeoutSeconds) {}

	void Reset(const RLGC::GameState& initialState) override {
		elapsed = 0.f;
	}

	bool IsTerminal(const RLGC::GameState& currentState) override {
		elapsed += currentState.deltaTime;
		return elapsed >= matchTimeout;
	}

	bool IsTruncation() override {
		return true;
	}
};

}
