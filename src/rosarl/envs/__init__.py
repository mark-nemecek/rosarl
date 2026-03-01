import copy

from gymnasium.envs.registration import WrapperSpec
from safety_gymnasium import __register_helper, register

import rosarl.envs.omnisafe_env
import rosarl.tasks

PREFIX = "Safetyterminal"
VERSION = "v0"
ROBOT_NAMES = ("Point", "Car", "Doggo", "Racecar", "Ant")


def __combine(
    prefix: str,
    tasks: dict[str, dict],
    agents: tuple[str],
    max_episode_steps: int,
    unsafe_terminal: bool = True,
    goal_terminal: bool = False,
):
    """Combine tasks and agents together to register environment tasks."""
    for task_name, task_config in tasks.items():
        # Vector inputs
        for robot_name in agents:
            env_id = f"{prefix}{robot_name}{task_name}-{VERSION}"
            combined_config = copy.deepcopy(task_config)
            combined_config.update({"agent_name": robot_name})

            wrapper = WrapperSpec(
                "RosarlWrapper",
                "rosarl.envs.wrappers:RosarlWrapper",
                {"unsafe_terminal": unsafe_terminal, "goal_terminal": goal_terminal},
            )

            __register_helper(
                env_id=env_id,
                entry_point="safety_gymnasium.builder:Builder",
                spec_kwargs={"config": combined_config, "task_id": env_id},
                max_episode_steps=max_episode_steps,
                additional_wrappers=[wrapper],
            )


# Button Environments
# ----------------------------------------
button_tasks = {"Button0": {}, "Button1": {}, "Button2": {}}
__combine(PREFIX, button_tasks, ROBOT_NAMES, max_episode_steps=1000)

# Push Environments
# ----------------------------------------
push_tasks = {"Push0": {}, "Push1": {}, "Push2": {}}
__combine(PREFIX, push_tasks, ROBOT_NAMES, max_episode_steps=1000)

# Goal Environments
# ----------------------------------------
goal_tasks = {"Goal0": {}, "Goal1": {}, "Goal2": {}}
__combine(PREFIX, goal_tasks, ROBOT_NAMES, max_episode_steps=1000)

# Circle Environments
# ----------------------------------------
circle_tasks = {"Circle0": {}, "Circle1": {}, "Circle2": {}}
__combine(PREFIX, circle_tasks, ROBOT_NAMES, max_episode_steps=500)

# Run Environments
# ----------------------------------------
run_tasks = {"Run0": {}}
__combine(PREFIX, run_tasks, ROBOT_NAMES, max_episode_steps=500)

# Pillar Environments
# ----------------------------------------
run_tasks = {
    "Pillar0": {},
    "Pillar1": {},
    "Pillar2": {},
    "Pillar3": {},
    "Pillar4": {},
    "Pillar5": {},
    "Pillar6": {},
    "Pillar7": {},
    "Pillar8": {},
}
__combine(PREFIX, run_tasks, ROBOT_NAMES, max_episode_steps=1000)


# Add versions which terminate on unsafe or goal
PREFIX = "Safetyterminalgoal"

# Button Environments
# ----------------------------------------
button_tasks = {"Button0": {}, "Button1": {}, "Button2": {}}
__combine(PREFIX, button_tasks, ROBOT_NAMES, max_episode_steps=1000, goal_terminal=True)

# Push Environments
# ----------------------------------------
push_tasks = {"Push0": {}, "Push1": {}, "Push2": {}}
__combine(PREFIX, push_tasks, ROBOT_NAMES, max_episode_steps=1000, goal_terminal=True)

# Goal Environments
# ----------------------------------------
goal_tasks = {"Goal0": {}, "Goal1": {}, "Goal2": {}}
__combine(PREFIX, goal_tasks, ROBOT_NAMES, max_episode_steps=1000, goal_terminal=True)

# Circle Environments
# ----------------------------------------
circle_tasks = {"Circle0": {}, "Circle1": {}, "Circle2": {}}
__combine(PREFIX, circle_tasks, ROBOT_NAMES, max_episode_steps=500, goal_terminal=True)

# Run Environments
# ----------------------------------------
run_tasks = {"Run0": {}}
__combine(PREFIX, run_tasks, ROBOT_NAMES, max_episode_steps=500, goal_terminal=True)

# Pillar Environments
# ----------------------------------------
run_tasks = {
    "Pillar0": {},
    "Pillar1": {},
    "Pillar2": {},
    "Pillar3": {},
    "Pillar4": {},
    "Pillar5": {},
    "Pillar6": {},
    "Pillar7": {},
    "Pillar8": {},
}
__combine(PREFIX, run_tasks, ROBOT_NAMES, max_episode_steps=1000, goal_terminal=True)


# Add versions which do not terminate on unsafe or goal
PREFIX = "Safetyterminalnone"

# Button Environments
# ----------------------------------------
button_tasks = {"Button0": {}, "Button1": {}, "Button2": {}}
__combine(
    PREFIX, button_tasks, ROBOT_NAMES, max_episode_steps=1000, unsafe_terminal=False
)

# Push Environments
# ----------------------------------------
push_tasks = {"Push0": {}, "Push1": {}, "Push2": {}}
__combine(
    PREFIX, push_tasks, ROBOT_NAMES, max_episode_steps=1000, unsafe_terminal=False
)

# Goal Environments
# ----------------------------------------
goal_tasks = {"Goal0": {}, "Goal1": {}, "Goal2": {}}
__combine(
    PREFIX, goal_tasks, ROBOT_NAMES, max_episode_steps=1000, unsafe_terminal=False
)

# Circle Environments
# ----------------------------------------
circle_tasks = {"Circle0": {}, "Circle1": {}, "Circle2": {}}
__combine(
    PREFIX, circle_tasks, ROBOT_NAMES, max_episode_steps=500, unsafe_terminal=False
)

# Run Environments
# ----------------------------------------
run_tasks = {"Run0": {}}
__combine(PREFIX, run_tasks, ROBOT_NAMES, max_episode_steps=500, unsafe_terminal=False)

# Pillar Environments
# ----------------------------------------
run_tasks = {
    "Pillar0": {},
    "Pillar1": {},
    "Pillar2": {},
    "Pillar3": {},
    "Pillar4": {},
    "Pillar5": {},
    "Pillar6": {},
    "Pillar7": {},
    "Pillar8": {},
}
__combine(PREFIX, run_tasks, ROBOT_NAMES, max_episode_steps=1000, unsafe_terminal=False)


# Humanoid Environment
wrapper = WrapperSpec(
    "RosarlWrapper",
    "rosarl.envs.wrappers:RosarlWrapper",
    {"unsafe_terminal": True, "goal_terminal": False},
)

__register_helper(
    env_id="SafetyterminalHalfCheetahVelocity-v1",
    entry_point="safety_gymnasium.tasks.safe_velocity.safety_half_cheetah_velocity_v1:SafetyHalfCheetahVelocityEnv",
    max_episode_steps=1000,
    reward_threshold=4800.0,
    additional_wrappers=[wrapper],
)

__register_helper(
    env_id="SafetyterminalHopperVelocity-v1",
    entry_point="safety_gymnasium.tasks.safe_velocity.safety_hopper_velocity_v1:SafetyHopperVelocityEnv",
    max_episode_steps=1000,
    reward_threshold=3800.0,
    additional_wrappers=[wrapper],
)

__register_helper(
    env_id="SafetyterminalSwimmerVelocity-v1",
    entry_point="safety_gymnasium.tasks.safe_velocity.safety_swimmer_velocity_v1:SafetySwimmerVelocityEnv",
    max_episode_steps=1000,
    reward_threshold=360.0,
    additional_wrappers=[wrapper],
)

__register_helper(
    env_id="SafetyterminalWalker2dVelocity-v1",
    entry_point="safety_gymnasium.tasks.safe_velocity.safety_walker2d_velocity_v1:SafetyWalker2dVelocityEnv",
    max_episode_steps=1000,
    additional_wrappers=[wrapper],
)

__register_helper(
    env_id="SafetyterminalAntVelocity-v1",
    entry_point="safety_gymnasium.tasks.safe_velocity.safety_ant_velocity_v1:SafetyAntVelocityEnv",
    max_episode_steps=1000,
    reward_threshold=6000.0,
    additional_wrappers=[wrapper],
)

__register_helper(
    env_id="SafetyterminalHumanoidVelocity-v1",
    entry_point="safety_gymnasium.tasks.safe_velocity.safety_humanoid_velocity_v1:SafetyHumanoidVelocityEnv",
    max_episode_steps=1000,
    additional_wrappers=[wrapper],
)
