import torch.nn as nn


class BasePolicy(nn.Module):
    """
    Base class for all policies.
    """

    def __init__(self):
        """
        Initialize the policy.
        """
        super(BasePolicy, self).__init__()
        pass

    def act(self, state):
        """
        Get action from the policy given the state.
        """
        raise NotImplementedError("act method not implemented")
