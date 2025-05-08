class Trajectory:
    """
    Class to store trajectory data.
    """

    def __init__(self, states, actions=None, rewards=None, next_states=None, done=None):
        self.states = states
        self.actions = actions
        self.done = done

    def __len__(self):
        return len(self.states)
