# Q-Function
import chainer
import chainer.functions as F
import chainer.links as L
import chainerrl

class QFunction(chainer.Chain):
    def __init__(self, obs_size, n_actions, n_hidden_channels):
        super().__init__(
            l0=L.Linear(obs_size, n_hidden_channels),
            l1=L.Linear(n_hidden_channels, n_hidden_channels),
            l2=L.Linear(n_hidden_channels, n_actions))

    def __call__(self, x, test=False):
        h = F.leaky_relu(self.l0(x))
        h = F.leaky_relu(self.l1(h))
        act = chainerrl.action_value.DiscreteActionValue(self.l2(h))
        return act
