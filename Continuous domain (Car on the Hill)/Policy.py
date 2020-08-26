import numpy as np

class Policy():
    """
    Class implementing a stationnary policy
    """

    def __init__(self, policyMatrix=None, model=None, useModel=False):
        # Matrix associating an action to each state
        self.matrix = policyMatrix

        # Model trained using fitted-Q-iteration
        self.model = model
        
        self.useModel = useModel

    def get_action(self, state):
        """
        Returns the action associated to a given state

        Arguments:
        ----------
        - `state`: (i,j) pair representing a state
          (N.B.: The (0,0) position is the top left corner of the grid)
        """
        if self.useModel == True:
            rewardBackward = self.model.predict((np.asarray([state[0], state[1], -4])).reshape(1, -1))
            rewardForward = self.model.predict((np.asarray([state[0], state[1], 4])).reshape(1, -1))
            
            if rewardBackward > rewardForward:
                return -4
            else:
                return 4
            
        else:
            return self.matrix[state[0]][state[1]]
