import gym
import time
import keyboard

ACTION = {
    "NOOP":         0,
    "RIGHT":        ["j", 1],
    "LEFT":         ["f", 2],
    "RFIRE":        ["k", 3],
    "LFIRE":        ["d", 4],
    "WORKAROUND":   ["l", 5]
}

def controller():
    if keyboard.is_pressed(ACTION["RIGHT"][0]):
        return ACTION["RIGHT"][1]
    if keyboard.is_pressed(ACTION["LEFT"][0]):
        return ACTION["LEFT"][1]
    if keyboard.is_pressed(ACTION["RFIRE"][0]):
        return ACTION["RFIRE"][1]
    if keyboard.is_pressed(ACTION["LFIRE"][0]):
        return ACTION["LFIRE"][1]
    return ACTION["NOOP"]


env = gym.make('SpaceInvaders-v0')
for match in range(50):
    observation = env.reset()
    done = False

    while (not done):
        env.render()
        action = controller()
        # action = env.action_space.sample()
        observation, reward, done, info = env.step(action)
        time.sleep(0.1)

env.close()