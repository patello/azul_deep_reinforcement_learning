from azulnet.agent import Agent
from azulnet.nn_runner import NNRunner

if __name__ == "__main__":
    agent = Agent(base_net_file=None)
    opponent = Agent(base_net_file=None)
    nnrunner = NNRunner(agent,opponent)
    nnrunner.run_batch(100)