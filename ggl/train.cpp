#include <GigaLearnCPP/Learner.h>

#include <RLGymCPP/Rewards/CommonRewards.h>
#include <RLGymCPP/Rewards/ZeroSumReward.h>
#include <RLGymCPP/TerminalConditions/NoTouchCondition.h>
#include <RLGymCPP/TerminalConditions/GoalScoreCondition.h>
#include <RLGymCPP/StateSetters/KickoffState.h>
#include <RLGymCPP/StateSetters/CombinedState.h>
#include <RLGymCPP/ObsBuilders/DefaultObs.h>
#include <RLGymCPP/ActionParsers/DefaultAction.h>

#include "rewards/CustomRewards.h"
#include "mutators/RandomFieldState.h"
#include "mutators/DrillTimeoutCondition.h"
#include "StepCallback.h"

#include <cstdlib>
#include <filesystem>
#include <iostream>
#include <string>

using namespace GGL;
using namespace RLGC;

static std::vector<WeightedReward> BuildRewards() {
	return {
		{ new AirReward(), 0.25f },

		{ new FaceBallReward(), 0.05f },
		{ new VelocityPlayerToBallReward(), 3.0f },
		{ new StrongTouchReward(20, 100), 25.f },

		{ new ZeroSumReward(new VelocityBallToGoalReward(), 1.f), 2.0f },

		{ new PickupBoostReward(), 10.f },
		{ new SaveBoostReward(), 0.2f },

		{ new ZeroSumReward(new BumpReward(), 0.5f), 20.f },
		{ new ZeroSumReward(new DemoReward(), 0.5f), 80.f },

		{ new GoalReward(), 250.f },

		// ported, unverified
		// { new GoalReward(-0.75f), 100.f },
		// { new rl_rl::AcceleratedTouchReward(0.2f, 1.0f), 4.0f },
		// { new rl_rl::InAirReward(), 0.4f },
		// { new rl_rl::DemoReward(0.5f), 70.0f },
		// { new rl_rl::SpeedBumpReward(), 15.0f },
		// { new ShotReward(), 15.0f },
		// { new SaveReward(), 20.0f },
		// { new PickupBoostReward(), 15.0f },
		// { new rl_rl::SupersonicSpeedReward(), 0.5f },
		// { new rl_rl::DeltaDistToBallReward(), 5.0f },
		// { new rl_rl::DeltaBallToGoalReward(), 5.0f },
	};
}

EnvCreateResult EnvCreateFunc(int index) {
	EnvCreateResult result = {};

	auto arena = Arena::Create(GameMode::SOCCAR);
	arena->AddCar(Team::BLUE);
	arena->AddCar(Team::ORANGE);
	result.arena = arena;

	result.stateSetter = new CombinedState({
		{ new KickoffState(), 0.60f },
		{ new rl_rl::RandomFieldState(), 0.40f },
	});

	result.terminalConditions = {
		new GoalScoreCondition(),
		new NoTouchCondition(30.f),
		new rl_rl::DrillTimeoutCondition(300.f),
	};

	result.obsBuilder = new DefaultObs();
	result.actionParser = new DefaultAction();
	result.rewards = BuildRewards();
	return result;
}

int main(int argc, char* argv[]) {
	const char* meshPath = std::getenv("GGL_MESH_DIR");
	if (!meshPath || !*meshPath) meshPath = "./collision_meshes";
	std::cout << "[rl_rl_ggl] Loading collision meshes from: " << meshPath << std::endl;
	RocketSim::Init(meshPath);

	LearnerConfig cfg = {};

	cfg.tickSkip = 8;
	cfg.actionDelay = cfg.tickSkip - 1;

	const char* numGamesEnv = std::getenv("GGL_NUM_GAMES");
	cfg.numGames = numGamesEnv ? std::atoi(numGamesEnv) : 36;

	cfg.randomSeed = -1;

	const int64_t tsPerItr = 100'000;
	cfg.ppo.tsPerItr = tsPerItr;
	cfg.ppo.batchSize = tsPerItr;
	cfg.ppo.miniBatchSize = tsPerItr;
	cfg.ppo.epochs = 1;

	cfg.ppo.policyLR = 2e-4f;
	cfg.ppo.criticLR = 2e-4f;
	cfg.ppo.gaeGamma = 0.99f;
	cfg.ppo.gaeLambda = 0.95f;
	cfg.ppo.clipRange = 0.2f;

	cfg.ppo.entropyScale = 0.01f;

	cfg.ppo.policy.layerSizes = { 512, 512, 512 };
	cfg.ppo.critic.layerSizes = { 512, 512, 512 };
	cfg.ppo.sharedHead.layerSizes = {};
	cfg.ppo.policy.activationType = ModelActivationType::RELU;
	cfg.ppo.critic.activationType = ModelActivationType::RELU;
	cfg.ppo.policy.optimType = ModelOptimType::ADAM;
	cfg.ppo.critic.optimType = ModelOptimType::ADAM;
	cfg.ppo.policy.addLayerNorm = false;
	cfg.ppo.critic.addLayerNorm = false;

	cfg.deviceType = LearnerDeviceType::AUTO;

	const char* runNameEnv = std::getenv("GGL_RUN_NAME");
	std::string runName = runNameEnv ? runNameEnv : "default";
	cfg.checkpointFolder = std::filesystem::path("checkpoints_ggl") / runName;
	cfg.tsPerSave = 1'000'000;
	cfg.checkpointsToKeep = 8;

	cfg.savePolicyVersions = true;
	cfg.tsPerVersion = 25'000'000;
	cfg.maxOldVersions = 32;

	cfg.trainAgainstOldVersions = true;
	cfg.trainAgainstOldChance = 0.15f;

	cfg.skillTracker.enabled = true;
	cfg.skillTracker.updateInterval = 128;
	cfg.skillTracker.numArenas = 8;
	cfg.skillTracker.simTime = 30;
	cfg.skillTracker.maxSimTime = 180;
	cfg.skillTracker.ratingInc = 5;
	cfg.skillTracker.initialRating = 0;

	cfg.sendMetrics = std::getenv("GGL_NO_METRICS") == nullptr;
	cfg.addRewardsToMetrics = cfg.sendMetrics;
	cfg.metricsProjectName = "rlgym-learn";
	cfg.metricsGroupName = "1v1-training";
	cfg.metricsRunName = runName;

	cfg.renderMode = std::getenv("GGL_RENDER") != nullptr;
	if (const char* spEnv = std::getenv("GGL_RENDER_SPEED")) {
		float sp = std::atof(spEnv);
		if (sp > 0.001f) cfg.renderTimeScale = sp;
	}

	Learner* learner = new Learner(EnvCreateFunc, cfg, rl_rl::StepCallback);
	learner->Start();
	return EXIT_SUCCESS;
}
