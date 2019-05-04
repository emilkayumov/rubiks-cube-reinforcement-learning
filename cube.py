import gym
import numpy as np
import pycuber
import torch
from gym import spaces
from sklearn.preprocessing import OneHotEncoder, LabelEncoder

ACTIONS = ["L", "L'", "R", "R'", "U", "U'", "D", "D'", "F", "F'", "B", "B'"]


def get_state(cube):
    raw_state = (
        str(cube)
            .replace(' ', '')
            .replace('\n', '')
            .replace('[', '')
            .replace(']', ''))

    return np.array(list(
        raw_state[:9] +
        raw_state[9:12] + raw_state[21:24] + raw_state[33:36] +
        raw_state[12:15] + raw_state[24:27] + raw_state[36:39] +
        raw_state[15:18] + raw_state[27:30] + raw_state[39:42] +
        raw_state[18:21] + raw_state[30:33] + raw_state[42:45] +
        raw_state[-9:]))


example_cube = pycuber.Cube()
ohe = OneHotEncoder()
le = LabelEncoder()
ohe.fit(get_state(example_cube).reshape((-1, 1)))


def get_observation(state):
    return torch.FloatTensor(ohe.transform(
        state.reshape((-1, 1))).todense()).view(-1)


N_ACTION = len(ACTIONS)
FINAL_STATE = get_state(example_cube)
FINAL_OBSERVATION = get_observation(FINAL_STATE)
N_SPACE = len(FINAL_OBSERVATION)


def color_matching_reward(str_state):
    return np.mean(str_state == FINAL_STATE)


def complete_reward(str_state):
    return int(all(str_state == FINAL_STATE))


def get_shuffled_cube(steps):
    shuffled_cube = pycuber.Cube()
    random_actions = np.random.choice(ACTIONS, size=steps)
    shuffled_cube.perform_algo(random_actions)
    return shuffled_cube


def is_done(state):
    return all(state == FINAL_STATE)


class CubeEnv(gym.Env):
    def __init__(self, steps, reward_scale=1,
                 reward_function=color_matching_reward,
                 random_steps=False):

        self.action_space = spaces.Discrete(N_ACTION)
        self.observation_space = spaces.Box(0, 1, shape=[N_SPACE])

        self._cube = None
        self._steps = steps
        self._quality = 0

        self._reward_scale = reward_scale
        self._reward_function = reward_function
        self._random_steps = random_steps

        self.seed()
        self.reset()

    def seed(self, seed=None):
        np.random.seed(seed)
        return seed

    def step(self, action):
        self._cube.perform_step(ACTIONS[action])
        state = get_state(self._cube)
        return (
            self.get_observation(state),
            self.calc_reward(state),
            self.is_done(state),
            {})

    def calc_reward(self, state):
        new_quality = self._reward_function(state) * self._reward_scale
        reward = new_quality - self._quality
        self._quality = new_quality
        return reward

    def reset(self):
        if self._random_steps:
            self._cube = get_shuffled_cube(
                np.random.randint(1, self._steps + 1))
        else:
            self._cube = get_shuffled_cube(self._steps)

        state = get_state(self._cube)
        observation = self.get_observation(state)
        self.calc_reward(state)
        return observation

    @staticmethod
    def get_observation(state):
        return get_observation(state)

    @staticmethod
    def is_done(state):
        return is_done(state)
