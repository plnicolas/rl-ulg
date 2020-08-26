import numpy as np
import math
import Policy


class Domain():
    """
    Class implementing a domain, and the methods that do not
    depend on its setting (deterministic or stochastic).
    Setting-specific methods (make_move, compute_Jn, compute_r, compute_p)
    are implemented in the DeterministicDomain and StochasticDomain subclasses
    """

    def __init__(self, discount, rewardMatrix, initialState):
        """
        Arguments:
        ----------
        - `discount`: the discount factor
        - `rewardMatrix`: a n x m reward matrix
        - `initialState`: the initial state (x, y)
        """

        # Discount factor
        self.discount = discount
        # Reward signal
        self.reward = rewardMatrix

        # Dimensions of the domain
        self.n = rewardMatrix.shape[0]
        self.m = rewardMatrix.shape[1]

        # Initial state
        self.init = initialState

        # Current position (i, j)
        self.pos = self.init

        # Current time step
        self.t = 0

        # Trajectory
        self.trajectory = []

        # J_N matrices
        self.J_N = []

        # Q_N matrices
        self.Q_N = []
        
        # Q values (as computed by the Q-learning algorithm)
        self.Q = np.zeros((self.n, self.m), dtype=object) # Initialised at 0 everywhere
        for i in range(0, self.n):
            for j in range(0, self.m):
                self.Q[i][j] = [0, 0, 0, 0]
        
        self.experience = []

        self.estimated_R = []
        self.estimated_P = []

        # Expected cumulative reward
        self.score = 0

    def string_to_pair(self, direction):
        """
        Function which converts a direction string into an (i, j) pair

        Arguments:
        ----------
        - `direction`: a direction string ("UP", "DOWN", "LEFT" or "RIGHT")

        Return:
        -------
        - The corresponding (i, j) pair (cf. action space in problem statement)
        """

        if direction == "UP":
            (i, j) = (-1, 0)

        if direction == "DOWN":
            (i, j) = (1, 0)

        if direction == "LEFT":
            (i, j) = (0, -1)

        if direction == "RIGHT":
            (i, j) = (0, 1)

        return (i, j)

    def retrieve_policy(self, N):
        """
        Retrieve an optimal policy based on the Q_N functions.
        N.B.: the Q_N functions must have been computed beforehand
        (using the compute_Q, estimate_Q or Q_learning function)
        """
        n = self.n
        m = self.m

        policyMatrix = np.zeros((n, m), dtype=object)

        for i in range(0, n):
            for j in range(0, m):
                index = np.argmax((self.Q_N[N - 1])[i][j])
                policyMatrix[i][j] = self.__index_to_string(index)

        policy = Policy.Policy(policyMatrix)
        return policy

    def __index_to_string(self, index):
        """
        Function which converts an index into a direction string
        (only used in the retrieve_policy method)

        Arguments:
        ----------
        - `index`: an index, with a value between 0 and 3 (included)
        (cf. retrieve_policy method)

        Return:
        -------
        - The corresponding direction string ("UP", "DOWN", "LEFT" or "RIGHT")
        """
        if index == 0:
            return "UP"
        elif index == 1:
            return "DOWN"
        elif index == 2:
            return "LEFT"
        else:
            return "RIGHT"
        
    def __string_to_index(self, string):
        """
        Function which converts an index into a direction string
        (only used in the retrieve_policy method)

        Arguments:
        ----------
        - 'string': a direction string ("UP", "DOWN", "LEFT" or "RIGHT")
        (cf. retrieve_policy method)

        Return:
        -------
        - The corresponding index, with a value between 0 and 3 (included)
        """
        if string == "UP":
            return 0
        elif string == "DOWN":
            return 1
        elif string == "LEFT":
            return 2
        else:
            return 3

    def compute_Q(self, N):
        """
        Function which computes Q_N(x,u).

        Arguments:
        ----------
        - `N`: the number of iterations
        """
        # Clear the previous Q_N matrices, if any
        self.Q_N = []

        n = self.n
        m = self.m

        # Base case:
        # Q_0(x,u) = 0 for all states and actions
        matrix = np.zeros((n, m), dtype=object)

        for i in range(0, n):
            for j in range(0, m):
                matrix[i][j] = (0, 0, 0, 0)
        self.Q_N.append(matrix)

        # The successive Q_N matrices are built incrementally
        for t in range(1, N):

            # Q_N-1 matrix, that will be used to compute Q_N
            prev = self.Q_N[t - 1]
            rewards = np.zeros((n, m), dtype=object)

            # For each state x (of coordinates (u, v))
            for u in range(0, n):
                for v in range(0, m):

                    pos = (u, v)
                    actionValues = []
                    # For each action
                    for a in ("UP", "DOWN", "LEFT", "RIGHT"):
                        # Computes Q_N(x, u) using the reward matrix and the
                        # previously computed Q_N-1 matrix to avoid redundant
                        # computations
                        maxSum = 0
                        for i in range(0, n):
                            for j in range(0, m):
                                maxSum += self.compute_p((i, j),
                                                         pos, a) * max(prev[i][j])
                        value = self.compute_r(pos, a) + self.discount * maxSum
                        actionValues.append(value)

                    rewards[u][v] = actionValues
            # Add the J_N matrix to the list
            self.Q_N.append(rewards)

    def estimate_r(self, trajectory):
        """
        Function which estimates r(x,u) based on a trajectory.
        The r(x,u) values are computed one time for each
        position and action and stored in a matrix to avoid redundant computations.

        Arguments:
        ----------
        - `trajectory`: the trajectory to use
        """
        n = self.n
        m = self.m

        length = len(trajectory)
        rewards = np.zeros((n, m), dtype=object)
        # For every state x = (i, j)
        for i in range(0, n):
            for j in range(0, m):
                x = (i, j)
                actionValues = []
                # For every action u
                for u in ("UP", "DOWN", "LEFT", "RIGHT"):
                    k = 0
                    rSum = 0
                    count = 0
                    # Compute an approximate value of r(x, u)
                    while k < ((length - 1) / 3):
                        if((trajectory[3 * k][0] == x) and (trajectory[3 * k + 1][1] == u)):
                            rSum += trajectory[3 * k + 2][2]
                            count += 1
                        k += 1

                    if count > 0:
                        actionValues.append(rSum / count)
                    else:
                        actionValues.append(0)

                rewards[i][j] = actionValues

        self.estimated_R = rewards

    def estimate_p(self, trajectory):
        """
        Function which estimates p(xp|x,u) based on a trajectory.
        The p(xp|x,u) values are computed one time for each
        possible (xp, x, u) triplet and stored in a matrix to avoid redundant computations.

        Arguments:
        ----------
        - `trajectory`: the trajectory to use
        """

        n = self.n
        m = self.m

        xpMatrix = np.zeros((n, m), dtype=object)

        # For every state xp = (i, j)
        for i in range(0, n):
            for j in range(0, m):

                xp = (i, j)
                xMatrix = np.zeros((n, m), dtype=object)

                # For every state x = (u, v)
                for u in range(0, n):
                    for v in range(0, m):

                        x = (u, v)
                        probas = []

                        # For every action u
                        for a in ("UP", "DOWN", "LEFT", "RIGHT"):
                            length = len(trajectory)
                            k = 0
                            pSum = 0
                            count = 0
                            while k < ((length - 1) / 3):
                                if((trajectory[3 * k][0] == x) and (trajectory[3 * k + 1][1] == a)):
                                    pSum += (trajectory[3 * k + 3][0] == xp)
                                    count += 1
                                k += 1

                            if count > 0:
                                probas.append(pSum / count)
                            else:
                                probas.append(0)

                        xMatrix[u][v] = probas

                xpMatrix[i][j] = xMatrix

        self.estimated_P = xpMatrix

    def estimate_Q(self, trajectory, N):
        """
        Function which computes Q_N(x,u).

        Arguments:
        ----------
        - `N`: the number of iterations
        """
        # Clear the previous Q_N matrices, if any
        self.Q_N = []

        n = self.n
        m = self.m

        self.estimate_r(trajectory)
        self.estimate_p(trajectory)

        # Base case:
        # Q_0(x,u) = 0 for all states and actions
        matrix = np.zeros((n, m), dtype=object)

        for i in range(0, n):
            for j in range(0, m):
                matrix[i][j] = (0, 0, 0, 0)
        self.Q_N.append(matrix)

        # The successive Q_N matrices are built incrementally
        for t in range(1, N):

            # Q_N-1 matrix, that will be used to compute Q_N
            prev = self.Q_N[t - 1]
            rewards = np.zeros((n, m), dtype=object)

            # For each state x (of coordinates (u, v))
            for u in range(0, n):
                for v in range(0, m):

                    actionValues = []
                    # For each action
                    for a in range(0, 4):
                        # Computes Q_N(x, u) using the reward matrix and the
                        # previously computed Q_N-1 matrix to avoid redundant
                        # computations
                        maxSum = 0
                        for i in range(0, n):
                            for j in range(0, m):
                                maxSum += ((self.estimated_P[i][j])
                                           [u][v])[a] * max(prev[i][j])
                        value = (self.estimated_R[u][v])[
                            a] + self.discount * maxSum
                        actionValues.append(value)

                    rewards[u][v] = actionValues
            # Add the J_N matrix to the list
            self.Q_N.append(rewards)

    def Q_learning(self, N, alpha, epsilon, initState, expReplay=False, batchSize=10):
        """
        Function which computes Q_N(x,u) with the Q-learning algorithm.

        Arguments:
        ----------
        - `N`: the number of iterations
        """

        moves = ["UP", "DOWN", "LEFT", "RIGHT"]

        # Base case:
        x_k = initState
        
        qValues = self.Q
        
        # Initiate experience list
        experience = self.experience

        # Start iterating
        for t in range(1, N):
            # The epsilon parameter is used for the exploration-exploitation dilemna
            if np.random.uniform(0, 1) < epsilon:
                index = np.random.randint(0, 4) # Exploration
                u_k = moves[index]
            else:
                index = np.argmax(qValues[x_k[0]][x_k[1]]) # Exploitation
                u_k = self.__index_to_string(index) 

            [x_k, u_k, r_k, x_next] = self.make_move(x_k, u_k)
            #print([x_k, u_k, r_k, x_next])
            experience.append([x_k, u_k, r_k, x_next])
            
            if expReplay == True:
                if len(experience) >= batchSize:
                    k = 0
                    while k < batchSize:
                        sample = np.random.randint(0, batchSize)
                        tup = experience[sample]
                        
                        x = tup[0]
                        u = tup[1]
                        r = tup[2]
                        xnext = tup[3]
                        
                        [i, j] = [x[0], x[1]]
                        nextmax = max(qValues[xnext[0]][xnext[1]])  
                        
                        Q_k = (qValues[i][j])[self.__string_to_index(u)]
                        Q_k = (1 - alpha) * Q_k + alpha * (r + self.discount * nextmax)
                        
                        (qValues[i][j])[self.__string_to_index(u)] = Q_k
                        
                        k += 1
                        

            [i, j] = [x_k[0], x_k[1]]
            nextmax = max(qValues[x_next[0]][x_next[1]])
            
            Q_k = (qValues[i][j])[index]
            Q_k = (1 - alpha) * Q_k + alpha * (r_k + self.discount * nextmax)

            (qValues[i][j])[index] = Q_k

            x_k = x_next

        self.Q = qValues
        self.experience = experience
            
        
    def retrieve_policy_Q_learning(self):
        """
        Retrieve an optimal policy based on the Q_N functions.
        N.B.: the Q_N functions must have been computed beforehand
        (using the compute_Q, estimate_Q or Q_learning function)
        """
        n = self.n
        m = self.m

        policyMatrix = np.zeros((n, m), dtype=object)

        for i in range(0, n):
            for j in range(0, m):
                index = np.argmax((self.Q)[i][j])
                policyMatrix[i][j] = self.__index_to_string(index)

        policy = Policy.Policy(policyMatrix)
        return policy

class DeterministicDomain(Domain):
    """
    Class implementing the deterministic version of the domain,
    and the methods that are specific to it
    """

    def __init__(self, discount, rewardMatrix, initialState):
        """
        Arguments:
        ----------
        - `discount`: the discount factor
        - `rewardMatrix`: a n x m reward matrix
        - `initialState`: the initial state (x, y)
        """

        Domain.__init__(self, discount, rewardMatrix, initialState)

    def make_move(self, pos, direction):
        """
        Perform an action (i.e., make a move) in a given direction
        (N.B.: The (0,0) position is the top left corner of the grid)
        The current position and score are updated
        This function is primarily used to generate a trajectory to be used
        with estimate_r, estimate_p and estimate_Q

        Arguments:
        ----------
        - `pos`: a (i, j) position
        - `direction`: string containing a direction;
          either "UP", "DOWN", "LEFT" or "RIGHT"
        """

        # The direction string is converted to the appropriate (i,j) pair
        (i, j) = self.string_to_pair(direction)

        n = self.n
        m = self.m
        newPos = (min(max(pos[0] + i, 0), n - 1),
                  min(max(pos[1] + j, 0), m - 1))

        # Update score and position
        r = self.reward[newPos[0]][newPos[1]]
        self.score += math.pow(self.discount, self.t) * r

        self.pos = newPos
        self.t += 1

        # Update trajectory
        self.trajectory.append([pos, direction, r])

        return [pos, direction, r, newPos]

    def compute_Jn(self, policy, N):
        """
        Function which computes the J_N(x) functions recursively for each
        state x and for a given policy.
        The J_N(x) functions are computed incrementally and stored in the form
        of matrices in self.J_N; this allows to avoid redundant computations

        Arguments:
        ----------
        - `policy`: the policy to apply
        - `N`: the value of N

        """
        # Clear the previous J_N matrices, if any
        self.J_N = []

        n = self.n
        m = self.m

        # Base case:
        # J_0(x) = 0 for all states
        self.J_N.append(np.zeros((n, m)))

        # The successive J_N matrices are built incrementally
        for t in range(1, N):

            # J_N-1 matrix, that will be used to compute J_N
            prev = self.J_N[t - 1]
            rewards = np.zeros((n, m))

            # For each state x (of coordinates (u, v))
            for u in range(0, n):
                for v in range(0, m):

                    pos = (u, v)
                    # Retrieve the next move using the policy
                    nextMove = policy.get_action(pos)
                    # Translates it into an (i, j) pair
                    (i, j) = self.string_to_pair(nextMove)
                    # Compute the new position
                    (x, y) = (min(max(pos[0] + i, 0), n - 1),
                              min(max(pos[1] + j, 0), m - 1))
                    # Computes J_N(x) using the reward matrix and the
                    # previously computed J_N-1 matrix to avoid redundant
                    # computations
                    rewards[u][v] = self.reward[x][y] + \
                        self.discount * prev[x][y]

            # Add the J_N matrix to the list
            self.J_N.append(rewards)

    def compute_r(self, x, u):
        """
        Function which computes r(x,u).

        Arguments:
        ----------
        - `x`: the starting state
        - `u`: the action taken by the agent
        """
        n = self.n
        m = self.m

        (i, j) = self.string_to_pair(u)
        # Compute the new position
        (x_, y_) = (min(max(x[0] + i, 0), n - 1),
                    min(max(x[1] + j, 0), m - 1))

        # Deterministic domain; simply return r(x, u)
        return self.reward[x_][y_]

    def compute_p(self, xp, x, u):
        """
        Function which computes p(xp|x,u).

        Arguments:
        ----------
        - `xp`: the destination state
        - `x`: the starting state
        - `u`: the action taken by the agent
        """
        n = self.n
        m = self.m

        (i, j) = self.string_to_pair(u)
        # Compute the new position
        (x_, y_) = (min(max(x[0] + i, 0), n - 1),
                    min(max(x[1] + j, 0), m - 1))

        # Deterministic domain
        if xp == (x_, y_):
            return 1
        else:
            return 0


class StochasticDomain(Domain):
    """
    Class implementing the stochastic version of the domain,
    and the methods that are specific to it
    """

    def __init__(self, discount, beta, rewardMatrix, initialState):
        """
        Arguments:
        ----------
        - `discount`: the discount factor
        - `beta`: the beta parameter
        - `rewardMatrix`: a n x m reward matrix
        - `initialState`: the initial state (x, y)
        """

        Domain.__init__(self, discount, rewardMatrix, initialState)
        self.beta = beta

    def make_move(self, pos, direction):
        """
        Perform an action (i.e., make a move) in a given direction
        (N.B.: The (0,0) position is the top left corner of the grid)
        The current position and score are updated
        This function is primarily used to generate a trajectory to be used
        with estimate_r, estimate_p and estimate_Q

        Arguments:
        ----------
        - `pos`: a (i, j) position
        - `direction`: string containing a direction;
          either "UP", "DOWN", "LEFT" or "RIGHT"
        """

        # The direction string is converted to the appropriate (i,j) pair
        (i, j) = self.string_to_pair(direction)

        w = np.random.uniform(0, 1)

        if w < (1 - self.beta):
            n = self.n
            m = self.m
            pos = self.pos
            newPos = (min(max(pos[0] + i, 0), n - 1),
                      min(max(pos[1] + j, 0), m - 1))
        else:
            newPos = (0, 0)

        # Update score and position
        r = self.reward[newPos[0]][newPos[1]]
        self.score += math.pow(self.discount, self.t) * r

        self.pos = newPos
        self.t += 1

        # Update trajectory
        self.trajectory.append([pos, direction, r])

        return [pos, direction, r, newPos]

    def compute_Jn(self, policy, N):
        """
        Function which computes the J_N(x) functions recursively for each
        state x and for a given policy.
        The J_N(x) functions are computed incrementally and stored in the form
        of matrices in self.J_N; this allows to avoid redundant computations

        Arguments:
        ----------
        - `policy`: the policy to apply
        - `N`: the value of N

        """
        # Clear the previous J_N matrices, if any
        self.J_N = []

        n = self.n
        m = self.m

        # Base case:
        # J_0(x) = 0 for all states
        self.J_N.append(np.zeros((n, m)))

        # The successive J_N matrices are built incrementally
        for t in range(1, N):

            # J_N-1 matrix, that will be used to compute J_N
            prev = self.J_N[t - 1]
            rewards = np.zeros((n, m))

            # For each state x (of coordinates (u, v))
            for u in range(0, n):
                for v in range(0, m):

                    pos = (u, v)
                    # Retrieve the next move using the policy
                    nextMove = policy.get_action(pos)
                    # Translates it into an (i, j) pair
                    (i, j) = self.string_to_pair(nextMove)
                    # Compute the new position
                    (x, y) = (min(max(pos[0] + i, 0), n - 1),
                              min(max(pos[1] + j, 0), m - 1))
                    # Computes J_N(x) using the reward matrix and the
                    # previously computed J_N-1 matrix to avoid redundant computations
                    # Stochastic setting: the esperance can be computed with a
                    # simple weighted sum
                    rewards[u][v] = self.beta * (self.reward[0][0] + self.discount * prev[0][0]) + (
                        1 - self.beta) * (self.reward[x][y] + self.discount * prev[x][y])

            # Add the J_N matrix to the list
            self.J_N.append(rewards)

    def compute_r(self, x, u):
        """
        Function which computes r(x,u).

        Arguments:
        ----------
        - `x`: the starting state
        - `u`: the action taken by the agent
        """
        n = self.n
        m = self.m

        (i, j) = self.string_to_pair(u)
        # Compute the new position
        (x_, y_) = (min(max(x[0] + i, 0), n - 1),
                    min(max(x[1] + j, 0), m - 1))

        # Stochastic domain; esperance over the two possibilities
        return self.beta * self.reward[0][0] + \
            (1 - self.beta) * (self.reward[x_][y_])

    def compute_p(self, xp, x, u):
        """
        Function which computes p(xp|x,u).

        Arguments:
        ----------
        - `xp`: the destination state
        - `x`: the starting state
        - `u`: the action taken by the agent
        """
        n = self.n
        m = self.m

        (i, j) = self.string_to_pair(u)
        # Compute the new position
        (x_, y_) = (min(max(x[0] + i, 0), n - 1),
                    min(max(x[1] + j, 0), m - 1))

        # Stochastic domain
        if xp == (0, 0):
            if (x_, y_) == (0, 0):
                return 1
            else:
                return self.beta
        else:
            if xp == (x_, y_):
                return (1 - self.beta)
            else:
                return 0
