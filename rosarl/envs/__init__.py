import copy

from gymnasium.envs.registration import WrapperSpec
from safety_gymnasium import __register_helper, register

from rosarl.envs.omnisafe_env import SafetyterminalGymnasiumEnv

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
