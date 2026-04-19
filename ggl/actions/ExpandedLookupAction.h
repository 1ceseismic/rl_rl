#pragma once
#include <RLGymCPP/ActionParsers/ActionParser.h>

namespace rl_rl {

// 126-action expanded table: adds 36 diagonal flip combos on top of the
// 90-action DefaultAction (24 ground + 66 aerial). Unlocks diagonal flips,
// yaw-cancel stalls, and more precise aerial control during double jumps.
//
// Mirrors ExpandedLookupTableAction in action_parser.py. Changing the action
// count breaks checkpoint compatibility.
class ExpandedLookupAction : public RLGC::ActionParser {
public:
	std::vector<RLGC::Action> actions;
	std::vector<uint8_t> groundMask, airMask, jumpMask, boostMask;

	ExpandedLookupAction();

	RLGC::Action ParseAction(int index, const RLGC::Player& player, const RLGC::GameState& state) override {
		return actions[index];
	}

	int GetActionAmount() override {
		return (int)actions.size();
	}

	std::vector<uint8_t> GetActionMask(const RLGC::Player& player, const RLGC::GameState& state) override;
};

}
