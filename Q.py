from torch.nn import Module, Linear, functional
import pfrl


class QFunction(Module):
    def __init__(self, obs_size, n_actions, n_hidden_channels):
        super().__init__()
        self.l1=Linear(obs_size, n_hidden_channels)
        self.l2=Linear(n_hidden_channels, n_hidden_channels)
        self.l3=Linear(n_hidden_channels, n_actions)

    def forward(self, x):
        h = functional.relu(self.l1(x))
        h = functional.relu(self.l2(h))
        return pfrl.action_value.DiscreteActionValue(self.l3(h))
