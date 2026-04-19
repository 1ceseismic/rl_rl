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
#include <iostream>
#include <string>

using namespace GGL;
using namespace RLGC;

// Reward weights from train.py:TrackedCombinedReward.
// Preserving the Python values 1:1. Note that GGL's PPO normalizes returns
// (standardizeReturns=true by default) so absolute weight scale matters less
// than relative magnitudes.
static std::vector<WeightedReward> BuildRewards() {
	return {
		// ── Goal / concede (GoalRatioReward with bias=0.25 ↔ GoalReward(concedeScale=-0.75)) ──
		// GoalRatioReward returns +goal_reward on scoring, -goal_reward*(1-bias) on concede.
		// bias=0.25 → concede = -0.75. Matches GGL's GoalReward(concedeScale=-0.75).
		{ new GoalReward(-0.75f), 100.f },

		// ── Touch dynamics ──
		{ new rl_rl::AcceleratedTouchReward(0.2f, 1.0f), 4.0f },

		// ── Aerial shaping ──
		{ new rl_rl::InAirReward(), 0.4f },

		// ── Demos (zero-sum built-in bias baked in) ──
		{ new rl_rl::DemoReward(0.5f), 70.0f },

		// ── Bumps (zero-sum, speed-scaled) ──
		{ new rl_rl::SpeedBumpReward(), 15.0f },

		// ── Event counters ──
		// NOTE: Python rewards used rsim_stats integer counts (1.0 per event). GGL's
		// built-in event rewards fire on the tick the event occurs (also 1.0), so the
		// per-event magnitude matches. Accumulation over the episode also matches.
		{ new ShotReward(), 15.0f },
		{ new SaveReward(), 20.0f },
		// PickupBoostReward differs: Python = integer count, GGL = sqrt-normalized
		// boost delta. Per-pickup reward in GGL is ~0.58 for a 33-boost pad, ~1.0
		// for a 100-boost pad. If this ends up too weak, raise the weight.
		{ new PickupBoostReward(), 15.0f },

		// ── Speed (supersonic_time based) ──
		{ new rl_rl::SupersonicSpeedReward(), 0.5f },

		// ── Potential-based shaping ──
		{ new rl_rl::DeltaDistToBallReward(), 5.0f },
		{ new rl_rl::DeltaBallToGoalReward(), 5.0f },
	};
}

EnvCreateResult EnvCreateFunc(int index) {
	EnvCreateResult result = {};

	// 1v1 soccar (matches train.py:team_size=1, spawn_opponents=True)
	auto arena = Arena::Create(GameMode::SOCCAR);
	arena->AddCar(Team::BLUE);
	arena->AddCar(Team::ORANGE);
	result.arena = arena;

	// State setter: 60% kickoff, 40% random field (matches RandomMutator in train.py)
	result.stateSetter = new CombinedState({
		{ new KickoffState(), 0.60f },
		{ new rl_rl::RandomFieldState(), 0.40f },
	});

	// Terminal conditions: goal scored → terminal; no-touch 30s or match-timeout 300s → truncation.
	result.terminalConditions = {
		new GoalScoreCondition(),
		new NoTouchCondition(30.f),
		new rl_rl::DrillTimeoutCondition(300.f),
	};

	result.obsBuilder = new DefaultObs();       // GGL default coefficients + rotation-matrix encoding
	result.actionParser = new DefaultAction();  // GGL default 90-action lookup + built-in masking
	result.rewards = BuildRewards();
	return result;
}

int main(int argc, char* argv[]) {
	// Collision meshes — required by RocketSim::Init(). Point GGL_MESH_DIR at the
	// folder produced by RLArenaCollisionDumper. See README_GGL.md.
	const char* meshPath = std::getenv("GGL_MESH_DIR");
	if (!meshPath || !*meshPath) meshPath = "./collision_meshes";
	std::cout << "[rl_rl_ggl] Loading collision meshes from: " << meshPath << std::endl;
	RocketSim::Init(meshPath);

	LearnerConfig cfg = {};

	// ── Simulation cadence ──
	cfg.tickSkip = 8;                 // 8-tick action repeat ↔ train.py:action_repeat=8
	cfg.actionDelay = cfg.tickSkip - 1;

	// ── Parallelism ──
	// train.py machine config: szmchn=36, nyx=48. Use env var for per-machine tuning.
	const char* numGamesEnv = std::getenv("GGL_NUM_GAMES");
	cfg.numGames = numGamesEnv ? std::atoi(numGamesEnv) : 36;

	// ── Seed (random each run) ──
	cfg.randomSeed = -1;

	// ── PPO ──
	// Mirrors train.py hyperparameters. Some field names differ (ent_coef ↔ entropyScale).
	const int64_t tsPerItr = 100'000;
	cfg.ppo.tsPerItr = tsPerItr;
	cfg.ppo.batchSize = tsPerItr;        // batch_size=ts_per_iter in train.py
	cfg.ppo.miniBatchSize = tsPerItr;    // Python uses 1 minibatch — mirror that
	cfg.ppo.epochs = 1;                  // n_epochs=1 in train.py

	cfg.ppo.policyLR = 2e-4f;            // lr=2e-4 in train.py
	cfg.ppo.criticLR = 2e-4f;
	cfg.ppo.gaeGamma = 0.99f;
	cfg.ppo.gaeLambda = 0.95f;
	cfg.ppo.clipRange = 0.2f;

	// entropyScale != Python ent_coef. GGL's is the scale of NORMALIZED entropy,
	// so it doesn't need to track action count. 0.01 preserves intent; tune up if
	// policy collapses or down if too exploratory.
	cfg.ppo.entropyScale = 0.01f;

	// ── Network ── Python: DiscreteFF(512,512,512) actor + BasicCritic(512,512,512).
	// Set sharedHead to empty for parity (Python has no shared layers).
	cfg.ppo.policy.layerSizes = { 512, 512, 512 };
	cfg.ppo.critic.layerSizes = { 512, 512, 512 };
	cfg.ppo.sharedHead.layerSizes = {}; // disabled — matches Python's separate actor/critic
	cfg.ppo.policy.activationType = ModelActivationType::RELU;
	cfg.ppo.critic.activationType = ModelActivationType::RELU;
	cfg.ppo.policy.optimType = ModelOptimType::ADAM;
	cfg.ppo.critic.optimType = ModelOptimType::ADAM;
	cfg.ppo.policy.addLayerNorm = false;  // Python net had no LayerNorm
	cfg.ppo.critic.addLayerNorm = false;

	// ── Device ── AUTO picks CUDA if available, else CPU.
	cfg.deviceType = LearnerDeviceType::AUTO;

	// ── Checkpoints ──
	cfg.checkpointFolder = "checkpoints_ggl";
	cfg.tsPerSave = 1'000'000;
	cfg.checkpointsToKeep = 8;

	// ── Metrics (wandb via python_scripts/metric_receiver.py) ──
	// Set GGL_NO_METRICS=1 for smoke runs without wandb. The embedded Python
	// interpreter is selected at GGL build time; if it's the system Python
	// (not the project venv), wandb may not be installed there and
	// MetricSender::init() will abort the run.
	cfg.sendMetrics = std::getenv("GGL_NO_METRICS") == nullptr;
	cfg.addRewardsToMetrics = cfg.sendMetrics;  // reward component breakdown shipped automatically
	cfg.metricsProjectName = "rlgym-learn";  // preserve wandb project from train.py
	cfg.metricsGroupName = "1v1-training";
	const char* runNameEnv = std::getenv("GGL_RUN_NAME");
	cfg.metricsRunName = runNameEnv ? runNameEnv : "ggl-port";

	// ── Render ── Off for training. Set GGL_RENDER=1 for live rlviser.
	cfg.renderMode = std::getenv("GGL_RENDER") != nullptr;

	Learner* learner = new Learner(EnvCreateFunc, cfg, rl_rl::StepCallback);
	learner->Start();
	return EXIT_SUCCESS;
}
