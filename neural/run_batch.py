from neural.agent import Agent
from game.nn_runner import NNRunner

agent = Agent(base_net_file=None)
opponent = Agent(base_net_file=None)
nnrunner = NNRunner(agent,opponent)
nnrunner.run_batch(100)