class Policy():
    """
    Class implementing a stationnary policy
    """

    def __init__(self, policyMatrix):
        # Matrix associating an action to each state
        self.matrix = policyMatrix

        # Dimensions of the domain
        self.n = policyMatrix.shape[0]
        self.m = policyMatrix.shape[1]

    def get_action(self, state):
        """
        Returns the action associated to a given state

        Arguments:
        ----------
        - `state`: (i,j) pair representing a state
          (N.B.: The (0,0) position is the top left corner of the grid)
        """
        return self.matrix[state[0]][state[1]]
