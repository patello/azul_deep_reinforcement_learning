from azulnet.agent import Agent
from azulnet.nn_runner import NNRunner
from azulnet.game_runner import GameRunner

if __name__ == "__main__":
    agent = Agent(base_net_file=None)
    opponent = Agent(base_net_file=None)
    game_runner = GameRunner(opponent=opponent)
    nnrunner = NNRunner(agent,game_runner)
    nnrunner.run_batch(100)