import pytest
import numpy as np
from neural.model import *
from game.nn_runner import NNRunner

def test_model_forward():
    ac_net = ActorCritic(136,180)
    runner = NNRunner()
    state = Variable(torch.from_numpy(runner.get_state_flat()).float().unsqueeze(0))
    #Test that the correct thing gets returned from the forward pass
    value, policy_dist = ac_net.forward(state)
    assert type(value)==torch.Tensor
    assert type(policy_dist)==torch.Tensor
    assert policy_dist.detach().numpy().size == 180
    assert np.sum(policy_dist.detach().numpy().squeeze(0)) == pytest.approx(1)
    value, policy_dist = ac_net.forward(state)
    #Test that masking works ok
    mask = np.zeros((1,180),dtype='bool')
    mask[(0,1)]=True
    value, policy_dist = ac_net.forward(state,torch.from_numpy(mask))
    assert type(value)==torch.Tensor
    assert type(policy_dist)==torch.Tensor
    assert policy_dist[(0,1)]==1
    assert np.sum(policy_dist.detach().numpy().squeeze(0)) == pytest.approx(1)
    #Test that masking with only zeros raises an exception
    mask = np.zeros((1,180),dtype='bool')
    with pytest.raises(IllegalMask):
        ac_net.forward(state,torch.from_numpy(mask))
