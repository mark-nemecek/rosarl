import copy

from gymnasium.envs.registration import WrapperSpec
from rosarl.envs.omnisafe_env import SafetyterminalGymnasiumEnv
from safety_gymnasium import __register_helper, register

PREFIX = "Safetyterminal"
VERSION = "v0"
ROBOT_NAMES = ("Point", "Car", "Doggo", "Racecar", "Ant")


def __combine(tasks, agents, max_episode_steps):
    """Combine tasks and agents together to register environment tasks."""
    for task_name, task_config in tasks.items():
        # Vector inputs
        for robot_name in agents:
            env_id = f"{PREFIX}{robot_name}{task_name}-{VERSION}"
            combined_config = copy.deepcopy(task_config)
            combined_config.update({"agent_name": robot_name})

            wrapper = WrapperSpec(
                "TerminalUnsafeWrapper",
                "rosarl.envs.wrappers:TerminalUnsafeWrapper",
                {},
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
__combine(button_tasks, ROBOT_NAMES, max_episode_steps=1000)


# Push Environments
# ----------------------------------------
push_tasks = {"Push0": {}, "Push1": {}, "Push2": {}}
__combine(push_tasks, ROBOT_NAMES, max_episode_steps=1000)


# Goal Environments
# ----------------------------------------
goal_tasks = {"Goal0": {}, "Goal1": {}, "Goal2": {}}
__combine(goal_tasks, ROBOT_NAMES, max_episode_steps=1000)


# Circle Environments
# ----------------------------------------
circle_tasks = {"Circle0": {}, "Circle1": {}, "Circle2": {}}
__combine(circle_tasks, ROBOT_NAMES, max_episode_steps=500)


# Run Environments
# ----------------------------------------
run_tasks = {"Run0": {}}
__combine(run_tasks, ROBOT_NAMES, max_episode_steps=500)
