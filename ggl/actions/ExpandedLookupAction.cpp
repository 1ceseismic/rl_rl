#include "ExpandedLookupAction.h"

using namespace rl_rl;
using RLGC::Action;

ExpandedLookupAction::ExpandedLookupAction() {
	constexpr float R_B[] = { 0, 1 };
	constexpr float R_F[] = { -1, 0, 1 };

	// Ground (24 actions — matches original)
	for (float throttle : R_F) {
		for (float steer : R_F) {
			for (float boost : R_B) {
				for (float handbrake : R_B) {
					if (boost == 1 && throttle != 1)
						continue;
					// throttle or boost: if boosting, act as full throttle
					float effective_throttle = (throttle != 0) ? throttle : boost;
					actions.push_back({
						effective_throttle, steer, 0, steer, 0, 0, boost, handbrake
					});
				}
			}
		}
	}

	int numGroundActions = (int)actions.size();

	// Aerial — Python includes ALL pitch/yaw/roll/jump/boost combos (no yaw==0 filter),
	// giving 102 aerial actions (3*3*3*2*2 = 108 minus 6 duplicates when pitch=roll=jump=0).
	for (float pitch : R_F) {
		for (float yaw : R_F) {
			for (float roll : R_F) {
				for (float jump : R_B) {
					for (float boost : R_B) {
						// Duplicate with ground (pure throttle/boost handled there)
						if (pitch == 0 && roll == 0 && jump == 0)
							continue;
						float handbrake = (jump == 1 && (pitch != 0 || yaw != 0 || roll != 0)) ? 1.f : 0.f;
						// Mirrors Python: throttle = boost, steer = yaw
						actions.push_back({
							boost, yaw, pitch, yaw, roll, jump, boost, handbrake
						});
					}
				}
			}
		}
	}

	groundMask.resize(actions.size());
	airMask.resize(actions.size());
	jumpMask.resize(actions.size());
	boostMask.resize(actions.size());

	for (int i = 0; i < (int)actions.size(); i++) {
		Action& action = actions[i];

		if (action.jump) jumpMask[i] = true;
		if (action.boost) boostMask[i] = true;

		if (i < numGroundActions) {
			groundMask[i] = true;
			// Ground actions with yaw-cancel (steer but no handbrake mismatch) also valid in air
			if (action.throttle == action.boost && (action.yaw != 0) == (action.handbrake != 0)) {
				airMask[i] = true;
			}
		} else if (!action.jump) {
			airMask[i] = true;
		}
	}
}

std::vector<uint8_t> ExpandedLookupAction::GetActionMask(const RLGC::Player& player, const RLGC::GameState& state) {
	auto result = std::vector<uint8_t>(actions.size(), false);

	auto fnApplyMask = [&](const std::vector<uint8_t>& mask, bool add) {
		if (add) {
			for (size_t i = 0; i < actions.size(); i++) result[i] |= mask[i];
		} else {
			for (size_t i = 0; i < actions.size(); i++) result[i] &= ~mask[i];
		}
	};

	if (player.isOnGround) {
		fnApplyMask(groundMask, true);
	} else {
		fnApplyMask(airMask, true);
	}

	if (player.boost == 0)
		fnApplyMask(boostMask, false);

	bool isTurtled = player.worldContact.hasContact && player.worldContact.contactNormal.z > 0.9f;
	if (player.HasFlipOrJump() || isTurtled)
		fnApplyMask(jumpMask, true);

	return result;
}
