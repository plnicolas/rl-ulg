import numpy as np
from sklearn.linear_model import LinearRegression
from sklearn.linear_model import LogisticRegression
from sklearn.ensemble import ExtraTreesRegressor
from sklearn.neural_network import MLPRegressor
import math
import Policy


class Domain():
    """
    Class implementing a domain, and the methods that do not
    depend on its setting (deterministic or stochastic).
    Setting-specific methods (make_move, compute_Jn, compute_r, compute_p)
    are implemented in the DeterministicDomain and StochasticDomain subclasses
    """

    def __init__(self, discount):
        """
        Arguments:
        ----------
        - `discount`: the discount factor
        - `rewardMatrix`: a n x m reward matrix
        - `initialState`: the initial state (x, y)
        """

        # Discount factor
        self.discount = discount

        # Limit values
        self.goalP = 1
        self.maxS = 3

        # Discretized state space "dimensions"
        self.n = 9
        self.m = 9

        # Current state [p, s]
        self.state = [0, 1]

        # Current time step
        self.t = 0

        # Trajectory
        self.trajectory = []

        # J_N matrix
        self.J_N = []

        # Q_N functions
        self.Q_N = []

        self.experience = []

        # Expected cumulative reward
        self.score = 0

    def make_move(self, u):
        """
        Make a move using a given action u.

        Arguments:
        ----------
        - `u`: the action to take

        Return:
        -------
        - True if the next state is terminal, false otherwise
        """
        [p, s] = [self.state[0], self.state[1]]
        [nextPos, nextSpeed] = self._euler_integration(p, s, u)

        # Compute reward for the move
        if nextPos < -1 or abs(nextSpeed) > self.maxS:
            reward = -1
        elif (nextPos > self.goalP and abs(nextSpeed) < self.maxS):
            reward = 1
        else:
            reward = 0

        # Update state, time step and trajectory
        self.state = [nextPos, nextSpeed]
        self.t += 1
        self.trajectory.append([(p, s), u, reward, (nextPos, nextSpeed)])

        return self.is_terminal_state(self.state)

    def _euler_integration(self, p, s, u):
        """
        Computes the acceleration at a given time step using Euler's
        integration method.

        Arguments:
        ----------
        - `p`: the current position
        - `s`: the current speed
        - `u`: the action taken

        Return:
        -------
        - The acceleration for the next time step
        """
        h = 0.001
        p_ = p
        s_ = s
        for i in range(100):
            p_ += h * s_
            s_ += h * self._f(p_, s_, u)

        return [p_, s_]

    def _f(self, p, s, u):
        """
        Computes the acceleration at a given integration step for Euler's
        integration method, using the equation given in the statement.

        Arguments:
        ----------
        - `p`: the current position
        - `s`: the current speed
        - `u`: the action taken

        Return:
        -------
        - The acceleration for the integration step
        """
        der_H = self._der_Hill(p)
        der_2_H = self._der_2_Hill(p)

        firstTerm = u / (1 + math.pow(der_H, 2))
        secondTerm = (9.81 * der_H) / (1 + math.pow(der_H, 2))
        thirdTerm = (math.pow(s, 2) * der_H * der_2_H) / \
            (1 + math.pow(der_H, 2))

        return (firstTerm - secondTerm - thirdTerm)

    def _Hill(self, p):
        """
        Hill function.

        Arguments:
        ----------
        - `p`: the current position

        Return:
        -------
        - Hill(p)
        """
        if p < 0:
            return (math.pow(p, 2) + p)
        else:
            return (p / (math.sqrt(1 + 5 * math.pow(p, 2))))

    def _der_Hill(self, p):
        """
        First-order derivative of the Hill function.

        Arguments:
        ----------
        - `p`: the current position

        Return:
        -------
        - Hill'(p)
        """
        if p < 0:
            return (2 * p + 1)
        else:
            return (1 / math.pow((1 + 5 * math.pow(p, 2)), 3 / 2))

    def _der_2_Hill(self, p):
        """
        Second-order derivative of the Hill function.

        Arguments:
        ----------
        - `p`: the current position

        Return:
        -------
        - Hill''(p)
        """
        if p < 0:
            return 2
        else:
            return (-(15 * p) / (math.pow((1 + 5 * math.pow(p, 2)), 5 / 2)))

    def is_terminal_state(self, state):
        """
        Checks whether a state is terminal or not.

        Arguments:
        ----------
        - `state`: a (position, speed) state

        Return:
        -------
        - True if the state is terminal, false otherwise
        """
        if abs(state[0]) > self.goalP or abs(state[1]) > 3:
            return True
        else:
            return False

    def compute_Jn(self, policy, N):
        """
        Function which computes an approximation of the J_N(x) expected cumulative
        reward for each state x and for a given policy.
        In order to compute this function, the state-space is discretized.

        Arguments:
        ----------
        - `policy`: the policy to apply
        - `N`: the value of N

        """
        # Clear the previous J_N matrix, if any
        self.J_N = []

        n = self.n
        m = self.m

        rewards = np.zeros((n, m))

        # For each discrete "state cell"
        for u in range(0, n):
            for v in range(0, m):
                # Monte-Carlo principle: average cumulative reward over several
                # initial states to approximate
                rewardsSum = 0
                for k in range(0, 10):
                    (pos, speed) = self._generate_init_state(u, v)
                    reward = 0
                    t = 0
                    # Run the agent for N step, starting from this initial
                    # state
                    while t < N:
                        # Retrieve the next move using the policy
                        nextMove = policy.get_action((u, v))

                        # Compute the new state
                        [nextPos, nextSpeed] = self._euler_integration(
                            pos, speed, nextMove)

                        # Update the cumulative reward signal
                        if nextPos < -1 or abs(nextSpeed) > self.maxS:
                            reward += math.pow(self.discount, t) * -1
                        elif (nextPos > self.goalP and abs(nextSpeed) < self.maxS):
                            reward += math.pow(self.discount, t) * 1
                        else:
                            reward += math.pow(self.discount, t) * 0

                        # Update the current state
                        (pos, speed) = (nextPos, nextSpeed)
                        t += 1

                        # Stop if a terminal state is reached
                        if self.is_terminal_state((pos, speed)):
                            break

                    rewardsSum += reward

                rewards[u][v] = rewardsSum / 10

            self.J_N = rewards

    def _generate_init_state(self, i, j):
        """
        Generate an initial state for one of the "discretized cells".

        Arguments:
        ----------
        - `i`: the discretized position class
        - 'j': the discretized speed class

        Return:
        -------
        - A randomly generated initial state for the (i, j) discrete cell
        """
        # Position
        if i < 4:
            position = -1 * \
                np.random.uniform((0.11 + 0.22 * ((8 - i) % 5)),
                                  (0.11 + 0.22 * ((8 - i) % 5)) + 0.22)
        elif i == 4:
            position = np.random.uniform(-0.11, 0.11)
        else:
            position = np.random.uniform(
                (0.11 + 0.22 * (i % 5)), (0.11 + 0.22 * (i % 5)) + 0.22)

        # Speed
        if j < 4:
            speed = -1 * \
                np.random.uniform((0.33 + 0.67 * ((8 - j) % 5)),
                                  (0.33 + 0.67 * ((8 - j) % 5)) + 0.67)
        elif j == 4:
            speed = np.random.uniform(-0.33, 0.33)
        else:
            speed = np.random.uniform(
                (0.33 + 0.67 * (j % 5)), (0.33 + 0.67 * (j % 5)) + 0.67)

        return (position, speed)

    def fitted_Q_iteration(self, SLmethod, N):
        """
        Approximate the Q_N function using the fitted-Q-iteration
        algorithm. 

        Arguments:
        ----------
        - 'SLmethod': the SL method to use; 0 for LinearRegression,
                      1 for extra trees, 2 for neural networks
        - 'N': the number of iterations

        Return:
        -------
        - The Q_N function
        """
        self._generate_trajectories()
        
        k = 1

        if SLmethod == 0:
            model = LinearRegression(n_jobs=-1)
        elif SLmethod == 1:
            model = ExtraTreesRegressor(n_estimators=50, max_depth=None, n_jobs=-1)
        else:
            model = MLPRegressor()

        [X_ls, y_ls] = self._generate_init_training_set(self.trajectory)

        model.fit(X_ls, y_ls)
        self.Q_N.append(model)

        k += 1

        while k <= N:
            """
            if SLmethod == 0:
                model = LinearRegression()
            elif SLmethod == 1:
                model = ExtraTreesRegressor(n_estimators=100, max_depth=8)
            else:
                model = MLPRegressor()
            """
            y_ls = self._update_training_set_outputs(y_ls, self.trajectory)
            
            model.fit(X_ls, y_ls)
            self.Q_N.append(model)

            k += 1
            


    def _generate_init_training_set(self, trajectory):
        """
        Generate a training set from a trajectory.

        Arguments:
        ----------
        - 'trajectory': the trajectory to use

        Return:
        -------
        - 'LS_inputs': the training set inputs
        - 'LS_outputs': the training set outputs
        """
        length = len(trajectory)

        LS_inputs = []
        LS_outputs = []

        i = 0
        while i < length:
            pos = ((trajectory[i])[0])[0]
            speed = ((trajectory[i])[0])[1]
            action = (trajectory[i])[1]
            LS_inputs.append([pos, speed, action])
            LS_outputs.append((trajectory[i])[2])
            i += 1

        # Convert the lists to numpy arrays (for usage in scikit-learn)
        LS_inputs = np.asarray(LS_inputs)
        LS_outputs = np.asarray(LS_outputs)
        return [LS_inputs, LS_outputs]


    def _update_training_set_outputs(self, previous_outputs, trajectory):
        """
        Update the fitted-Q-iteration training set outputs.

        Arguments:
        ----------
        - 'previous_out': the training set outputs of the previous iteration
        - 'trajectory': the trajectory

        Return:
        -------
        - 'new_outputs': the updated outputs
        """ 
        print(len(previous_outputs))
        Q_prev = self.Q_N[len(self.Q_N)-1]

        new_outputs = []
        
        left = []
        right = []

        i = 0
        while i < len(trajectory):
            r_k = previous_outputs[i]
            pos_next = ((trajectory[i])[3])[0]
            speed_next = ((trajectory[i])[3])[1]
            right.append([pos_next, speed_next, 4])
            left.append([pos_next, speed_next, -4])
            i += 1
        
        right_values = (Q_prev.predict((np.asarray(right))))
        left_values = (Q_prev.predict((np.asarray(left))))
        
        i = 0
        while i < len(trajectory):
            maxQprev = max(right_values[i], left_values[i])
            out = r_k + self.discount * maxQprev
            new_outputs.append(out)
            i += 1

        new_outputs = np.ndarray.flatten(np.asarray(new_outputs))
        return new_outputs
    
    def _generate_trajectories(self):
        """
        Generate random trajectories to be used in the fitted-Q-iteration
        algorithm.
        """ 
        actions = [-4, 4]
        # Generate 100 trajectories
        for i in range(0, 200):
            u = np.random.randint(0,9)
            v = np.random.randint(0,9)
            (pos, speed) = self._generate_init_state(u, v)
            # Run the agent for at most 100 step, starting from this initial
            # state
            for t in range (0,100):
                # Select a move at random
                nextMove = np.random.choice(actions, 1)

                # Compute the new state
                [nextPos, nextSpeed] = self._euler_integration(
                    pos, speed, nextMove)

                # Compute the reward
                if nextPos < -1 or abs(nextSpeed) > self.maxS:
                    reward = -1
                elif (nextPos > self.goalP and abs(nextSpeed) < self.maxS):
                    reward = 1
                else:
                    reward = 0

                self.trajectory.append([(pos, speed), nextMove, reward, (nextPos, nextSpeed)])
                # Update the current state
                (pos, speed) = (nextPos, nextSpeed)

                # Stop if a terminal state is reached
                if self.is_terminal_state((pos, speed)):
                    break



