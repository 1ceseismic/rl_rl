import os

os.environ["OPENBLAS_NUM_THREADS"] = "1"
from rewards import VelocityPlayerToBallReward, InAirReward, FaceForwardReward, GoalRatioReward, SpeedReward, DenseSpeedReward, BoostManagementReward, TouchBallVelocityReward, OpponentProximityPenalty, FlipReward, FlipHitReward
from metrics import CustomMetricsProvider

def build_env():
    import numpy as np
    from rlgym.api import RLGym
    from rlgym.rocket_league import common_values
    from rlgym.rocket_league.action_parsers import LookupTableAction, RepeatAction
    from rlgym.rocket_league.done_conditions import (
        AnyCondition,
        GoalCondition,
        NoTouchTimeoutCondition,
        TimeoutCondition,
    )
    from rlgym.rocket_league.obs_builders import DefaultObs
    from rlgym.rocket_league.reward_functions import (
        CombinedReward,
        GoalReward,
        TouchReward,
    )
    from rlgym.rocket_league.rlviser import RLViserRenderer
    from rlgym.rocket_league.sim import RocketSimEngine
    from rlgym.rocket_league.state_mutators import (
        FixedTeamSizeMutator,
        KickoffMutator,
        MutatorSequence,
    )

    spawn_opponents = True
    team_size = 1
    blue_team_size = team_size
    orange_team_size = team_size if spawn_opponents else 0
    action_repeat = 8
    no_touch_timeout_seconds = 30
    game_timeout_seconds = 300

    action_parser = RepeatAction(LookupTableAction(), repeats=action_repeat)
    termination_condition = GoalCondition()
    truncation_condition = AnyCondition(
        NoTouchTimeoutCondition(timeout_seconds=no_touch_timeout_seconds),
        TimeoutCondition(timeout_seconds=game_timeout_seconds),
    )

    reward_fn = CombinedReward(
        (GoalRatioReward(), 10),
        (TouchReward(), 0.4),
        (VelocityPlayerToBallReward(), 0.4),
        (InAirReward(), 0.1),
        (FaceForwardReward(), 0.4),
        (SpeedReward(), 0.3),
        (DenseSpeedReward(), 0.05),
        (BoostManagementReward(), 0.05),
        (TouchBallVelocityReward(), 1.0),
        (OpponentProximityPenalty(), 0.1),
        (FlipReward(), 0.005),
        (FlipHitReward(), 2.0),
    )

    obs_builder = DefaultObs(
        zero_padding=team_size,
        pos_coef=np.asarray(
            [
                1 / common_values.SIDE_WALL_X,
                1 / common_values.BACK_NET_Y,
                1 / common_values.CEILING_Z,
            ]
        ),
        ang_coef=1 / np.pi,
        lin_vel_coef=1 / common_values.CAR_MAX_SPEED,
        ang_vel_coef=1 / common_values.CAR_MAX_ANG_VEL,
        boost_coef=1 / 100.0,
    )

    state_mutator = MutatorSequence(
        FixedTeamSizeMutator(blue_size=blue_team_size, orange_size=orange_team_size),
        KickoffMutator(),
    )
    return RLGym(
        state_mutator=state_mutator,
        obs_builder=obs_builder,
        action_parser=action_parser,
        reward_fn=reward_fn,
        termination_cond=termination_condition,
        truncation_cond=truncation_condition,
        transition_engine=RocketSimEngine(),
        shared_info_provider=CustomMetricsProvider(),
        renderer=RLViserRenderer(tick_rate=120/8),
    )


if __name__ == "__main__":
    from typing import Tuple

    import numpy as np
    from rlgym_learn_algos.logging import (
        WandbMetricsLogger,
        WandbMetricsLoggerConfigModel,
    )
    from rlgym_learn_algos.ppo import (
        BasicCritic,
        DiscreteFF,
        ExperienceBufferConfigModel,
        GAETrajectoryProcessor,
        GAETrajectoryProcessorConfigModel,
        NumpyExperienceBuffer,
        PPOAgentController,
        PPOAgentControllerConfigModel,
        PPOLearnerConfigModel,
    )
    from metrics import CustomMetricsLogger

    from rlgym_learn import (
        BaseConfigModel,
        LearningCoordinator,
        LearningCoordinatorConfigModel,
        NumpySerdeConfig,
        ProcessConfigModel,
        PyAnySerdeType,
        SerdeTypesModel,
        generate_config,
    )
    from rlgym_learn.rocket_league import GameStatePythonSerde

    # The obs_space_type and action_space_type are determined by your choice of ObsBuilder and ActionParser respectively.
    # The logic used here assumes you are using the types defined by the DefaultObs and LookupTableAction above.
    DefaultObsSpaceType = Tuple[str, int]
    DefaultActionSpaceType = Tuple[str, int]

    def actor_factory(
        obs_space: DefaultObsSpaceType,
        action_space: DefaultActionSpaceType,
        device: str,
    ):
        return DiscreteFF(obs_space[1], action_space[1], (256, 256, 256), device)

    def critic_factory(obs_space: DefaultObsSpaceType, device: str):
        return BasicCritic(obs_space[1], (256, 256, 256), device)

    import socket
    from checkpoint import select_checkpoint
    hostname = socket.gethostname()

    # Per-machine config: (n_proc, render)
    machine_config = {
        "szmhcn": (36, True),    # 24 cores, local with rlviser
        "nyx":    (48, False),   # 32 cores, headless
    }
    n_proc, render = machine_config.get(hostname, (36, False))
    timestep_limit = 1_000_000_000
    lr = 2e-4      #2e-4 until silver, 1e-4 after
    ts_per_iter = 50_000
    exp_buf_steps = ts_per_iter*3  #default is 200k

    checkpoint_load, run_name, parent_checkpoint = select_checkpoint()

    # Create the config that will be used for the run
    config = LearningCoordinatorConfigModel(
        base_config=BaseConfigModel(
            serde_types=SerdeTypesModel(
                agent_id_serde_type=PyAnySerdeType.STRING(),
                action_serde_type=PyAnySerdeType.NUMPY(np.int64),
                obs_serde_type=PyAnySerdeType.NUMPY(np.float64),
                reward_serde_type=PyAnySerdeType.FLOAT(),
                obs_space_serde_type=PyAnySerdeType.TUPLE(
                    (PyAnySerdeType.STRING(), PyAnySerdeType.INT())
                ),
                action_space_serde_type=PyAnySerdeType.TUPLE(
                    (PyAnySerdeType.STRING(), PyAnySerdeType.INT())
                ),
                shared_info_serde_type=PyAnySerdeType.DYNAMIC(),
            ),
            timestep_limit=timestep_limit,
        ),
        process_config=ProcessConfigModel(
            n_proc=n_proc,
            render=render,
            render_delay=8/120,  #8/120 is default; for realtime (but halts rest of processes for learning by ~67ms)
        ),
        agent_controllers_config={
            "PPO1": PPOAgentControllerConfigModel(
                run_name=run_name,
                add_unix_timestamp=False,
                checkpoint_load_folder=checkpoint_load,
                learner_config=PPOLearnerConfigModel(
                    ent_coef=0.01,
                    actor_lr=lr,
                    critic_lr=lr,
                    batch_size=ts_per_iter #ts per iteration,    use this * 2 or 3 for exp buffer
                ),
                experience_buffer_config=ExperienceBufferConfigModel(
                    max_size=exp_buf_steps,
                    trajectory_processor_config=GAETrajectoryProcessorConfigModel(),
                ),
                metrics_logger_config=WandbMetricsLoggerConfigModel(
                    group="1v1-training",
                    run=run_name,
                    settings_kwargs={"entity": "rl_rlbot"},
                    additional_wandb_run_config={"parent_checkpoint": parent_checkpoint},
                ),
            )
        },
        agent_controllers_save_folder="agent_controllers_checkpoints",  # (default value) WARNING: THIS PROCESS MAY DELETE ANYTHING INSIDE THIS FOLDER. This determines the parent folder for the runs for each agent controller. The runs folder for the agent controller will be this folder and then the agent controller config key as a subfolder.
    )

    # Generate the config file for reference (this file location can be
    # passed to the learning coordinator via config_location instead of defining
    # the config object in code and passing that)
    generate_config(
        learning_coordinator_config=config,
        config_location="config.json",
        force_overwrite=True,
    )

    learning_coordinator = LearningCoordinator(
        build_env,
        agent_controllers={
            "PPO1": PPOAgentController(
                actor_factory=actor_factory,
                critic_factory=critic_factory,
                experience_buffer=NumpyExperienceBuffer(GAETrajectoryProcessor()),
                metrics_logger=WandbMetricsLogger(CustomMetricsLogger()),
                obs_standardizer=None,
            )
        },
        config=config,
    )
    learning_coordinator.start()