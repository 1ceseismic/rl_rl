#pragma once
#include <RLGymCPP/TerminalConditions/TerminalCondition.h>

namespace rl_rl {

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
