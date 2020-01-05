import gym
from neural.agent import Agent
from game.nn_runner import NNRunner
import sys

agent = Agent(base_net_file=None)
nnrunner = NNRunner(agent)
nnrunner.run_batch(100)