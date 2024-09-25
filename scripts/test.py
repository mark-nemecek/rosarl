from rosarl import envs
import safety_gymnasium as sg


# env = sg.make("SafetyterminalPointGoal0-v0")
# print(env)

# env_id = "SafetyterminalPointGoal1-v0"
# env_id = "SafetyterminalPointPush1-v0"
env_id = "SafetyterminalPointButton1-v0"
env = sg.make(env_id, render_mode="human")

obs, info = env.reset()
while True:
    act = env.action_space.sample()
    for _ in range(10):
        obs, reward, cost, terminated, truncated, info = env.step(act)
        if terminated or truncated:
            break
    if terminated or truncated:
            break
    env.render()
