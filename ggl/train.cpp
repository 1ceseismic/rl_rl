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

// Default reward set: copied from GigaLearnCPP's ExampleMain.cpp, which claims
// "produces a scoring bot in ~100m steps." Starting from a known-good GGL
// baseline before layering on project-specific shaping.
//
// The Python ports (AcceleratedTouch, InAir, Demo, SpeedBump, SupersonicSpeed,
// DeltaDistToBall, DeltaBallToGoal) live in ggl/rewards/CustomRewards.h and
// can be re-enabled by uncommenting the block below — but they're known to
// be rough ports and should not be trusted before re-auditing.
static std::vector<WeightedReward> BuildRewards() {
	// Phase 2 (applied @ ~500M steps): peel back bootstrap shaping, lean into
	// the sparse goal signal now that basic ball interaction is learned.
	// Previous (Phase 1 / GGL default) values are in the trailing comments.
	// If training regresses, revert individual weights and continue from the
	// branched checkpoint.
	return {
		// Movement — keep phase-1 weight
		{ new AirReward(), 0.25f },

		// Player-ball — peel back Face, keep some ball-seeking
		{ new FaceBallReward(), 0.05f },                                   // was 0.25
		{ new VelocityPlayerToBallReward(), 3.0f },                        // was 4.0
		{ new StrongTouchReward(20, 100), 25.f },                          // was 60

		// Ball-goal — purposeful, keep
		{ new ZeroSumReward(new VelocityBallToGoalReward(), 1.f), 2.0f },  // unchanged

		// Boost — unchanged, economy still matters
		{ new PickupBoostReward(), 10.f },
		{ new SaveBoostReward(), 0.2f },

		// Game events — unchanged; already sparse so they won't dominate
		{ new ZeroSumReward(new BumpReward(), 0.5f), 20.f },
		{ new ZeroSumReward(new DemoReward(), 0.5f), 80.f },

		// The real objective — make it more dominant
		{ new GoalReward(), 250.f },                                       // was 150

		// ── Project-specific rewards ported from rewards.py (DISABLED) ────
		// Enable once individually re-verified against the Python source.
		// { new GoalReward(-0.75f), 100.f },                     // GoalRatioReward(bias=0.25)
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
	// Per-run layout: checkpoints_ggl/<run-name>/<timestep>/ so multiple runs can
	// coexist and the Python picker can list them. GGL_RUN_NAME below feeds both
	// the wandb metrics and the checkpoint folder path.
	const char* runNameEnv = std::getenv("GGL_RUN_NAME");
	std::string runName = runNameEnv ? runNameEnv : "default";
	cfg.checkpointFolder = std::filesystem::path("checkpoints_ggl") / runName;
	cfg.tsPerSave = 1'000'000;
	cfg.checkpointsToKeep = 8;

	// ── Policy versioning + ELO skill tracking ──
	// Saves a policy snapshot every tsPerVersion steps; the SkillTracker uses
	// those snapshots to play head-to-head matches against the current policy
	// and logs Rating/<mode> (e.g. Rating/1v1) to wandb. Ratings will be flat
	// until the bot starts scoring reliably (~100M), but turning this on early
	// is harmless — snapshots accumulate so the graph is populated when matches
	// finally produce signal.
	cfg.savePolicyVersions = true;
	cfg.tsPerVersion = 25'000'000;     // snapshot every 25M steps
	cfg.maxOldVersions = 32;

	cfg.trainAgainstOldVersions = true;      // 15% of rollouts vs a random old version
	cfg.trainAgainstOldChance = 0.15f;

	cfg.skillTracker.enabled = true;
	// updateInterval: rating matches pause the collection thread while they
	// run on numArenas for up to maxSimTime. At tsPerItr=100k, the previous
	// value of 16 meant a pause every ~1.6M steps — the collection-rate
	// chart on wandb showed periodic dips to 30k/0 sps. 128 pushes that out
	// to every ~12.8M steps; still populates the Rating/<mode> series, but
	// doesn't dominate wall-clock.
	cfg.skillTracker.updateInterval = 128;
	cfg.skillTracker.numArenas = 8;          // was 16 — half the eval-time compute
	cfg.skillTracker.simTime = 30;           // was 45 — shorter match, less pause
	cfg.skillTracker.maxSimTime = 180;       // was 240
	cfg.skillTracker.ratingInc = 5;
	cfg.skillTracker.initialRating = 0;      // relative system; offset in wandb if desired

	// ── Metrics (wandb via python_scripts/metric_receiver.py) ──
	// Set GGL_NO_METRICS=1 for smoke runs without wandb. The embedded Python
	// interpreter is selected at GGL build time; if it's the system Python
	// (not the project venv), wandb may not be installed there and
	// MetricSender::init() will abort the run.
	cfg.sendMetrics = std::getenv("GGL_NO_METRICS") == nullptr;
	cfg.addRewardsToMetrics = cfg.sendMetrics;  // reward component breakdown shipped automatically
	cfg.metricsProjectName = "rlgym-learn";  // preserve wandb project from train.py
	cfg.metricsGroupName = "1v1-training";
	cfg.metricsRunName = runName;

	// ── Render ── Off for training. Set GGL_RENDER=1 for live rlviser.
	// GGL_RENDER_SPEED sets the initial speed multiplier (e.g., 5.0 = 5x real
	// time). Each frame the UI slider overrides this via get_game_speed() in
	// render_receiver.py.
	cfg.renderMode = std::getenv("GGL_RENDER") != nullptr;
	if (const char* spEnv = std::getenv("GGL_RENDER_SPEED")) {
		float sp = std::atof(spEnv);
		if (sp > 0.001f) cfg.renderTimeScale = sp;
	}

	Learner* learner = new Learner(EnvCreateFunc, cfg, rl_rl::StepCallback);
	learner->Start();
	return EXIT_SUCCESS;
}
