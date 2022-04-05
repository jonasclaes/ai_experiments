import gym
import random
import numpy as np
from tensorflow.keras.models import Sequential
from tensorflow.keras.layers import Dense, Flatten
from tensorflow.keras.optimizers import Adam

from rl.agents import DQNAgent
from rl.policy import BoltzmannQPolicy
from rl.memory import SequentialMemory


def build_model(states, actions):
    model = Sequential()
    model.add(Flatten(input_shape=(1,states)))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(24, activation='relu'))
    model.add(Dense(actions, activation='linear'))
    return model

def build_agent(model, actions):
    policy = BoltzmannQPolicy()
    memory = SequentialMemory(limit=50000, window_length=1)
    dqn = DQNAgent(model=model, memory=memory, policy=policy, 
                  nb_actions=actions, nb_steps_warmup=10, target_model_update=1e-1)
    return dqn

env = gym.make('CartPole-v1')
states = env.observation_space.shape[0]
actions = env.action_space.n

model = build_model(states, actions)

dqn = build_agent(model, actions)
dqn.compile(Adam(lr=1e-2), metrics=['mae'])
dqn.load_weights('dqn_weights.h5f')

_ = dqn.test(env, nb_episodes=10, visualize=True)
