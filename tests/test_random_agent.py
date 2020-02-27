import os

from azulnet.game_runner import *
from azulnet import Azul

#Script dir as per suggestion here: https://stackoverflow.com/questions/7165749/open-file-in-a-relative-location-in-python
script_dir = os.path.dirname(__file__)

def test_random_agent_get_ac_output():
    opponent=RandomAgent()
    game=Azul()
    game.import_JSON(script_dir+"/resources/game_first_round.json")
    all_valid = check_all_valid(game)
    #Make a number of random moves and check that they are between 0..180 and that they are valid
    moves=np.zeros(1000,dtype="int")
    for i in range(moves.size):
        moves[i]=opponent.get_a_output(None,torch.from_numpy(all_valid.reshape(1,180)))
        assert all_valid[moves[i]]
    assert np.count_nonzero(moves > 179) == 0
    assert np.count_nonzero(moves < 0) == 0