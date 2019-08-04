from stable_baselines.common.vec_env import DummyVecEnv, SubprocVecEnv

from environments.continuous_ear_env import EarEnv


if __name__ == "__main__":
    env = SubprocVecEnv([lambda: EarEnv()])

    obs = env.reset()

    for i in range(4):
        #print(env.observation_space.sample())
        if i == 0:
            action = [0.1, 1.0]
            #action = [1.2, 50.0]

        else:
            action = env.action_space.sample()
        obs, reward, done, info = env.step([action])
        print("obs: {}, action: {}, reward: {}, episode: {}".format(obs, action, reward, info))
