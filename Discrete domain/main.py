import numpy as np
import matplotlib.pyplot as plt
import math
import Domain
import Policy

# Domain instance (cf. assignement statement)
rewardMatrix = np.array([[-3.0, 1.0, -5.0, 0.0, 19.0],
                         [6.0, 3.0, 8.0, 9.0, 10.0],
                         [5.0, -8.0, 4.0, 1.0, -8.0],
                         [6.0, -9.0, 4.0, 19.0, -5.0],
                         [-20.0, -17.0, -4.0, -3.0, 9.0]])

print("Execution of section 6 to 7 of the assignement...")
print("This should take approximately 2 minutes.")

# Function to perform the whole assignement for a given discount factor
# N.B.: Section 2 to 5 (previous parts of the assignement were put inside
# comments to speed-up execution
def do_assignement(discount):
    """
    #########
    # SECTION 2: Implementation of the domain
    #########

    # Deterministic setting
    domain = Domain.DeterministicDomain(discount, rewardMatrix, (3, 2))
    for t in range(0, 10):
        domain.make_move(domain.pos, "UP")

    # Stochastic setting
    domain = Domain.StochasticDomain(discount, 0.5, rewardMatrix, (3, 2))
    for t in range(0, 10):
        domain.make_move(domain.pos, "UP")

    #########
    # SECTION 3: Expected return of a policy
    #########

    print("")
    print("Section 3:")
    print("")

    # Creation of the always-go-up policy (cf. assignment statement)
    policyMatrix = np.array([["UP", "UP", "UP", "UP", "UP"],
                             ["UP", "UP", "UP", "UP", "UP"],
                             ["UP", "UP", "UP", "UP", "UP"],
                             ["UP", "UP", "UP", "UP", "UP"],
                             ["UP", "UP", "UP", "UP", "UP"]])
    alwaysUp = Policy.Policy(policyMatrix)

    print("Deterministic domain:")
    print("")
    # Deterministic setting
    domain = Domain.DeterministicDomain(discount, rewardMatrix, (0, 0))
    domain.compute_Jn(alwaysUp, 1001)
    print(domain.J_N[1000])

    print("")
    print("")

    print("Stochastic domain (beta = 0.5):")
    print("")
    # Stochastic setting
    domain = Domain.StochasticDomain(discount, 0.5, rewardMatrix, (0, 0))
    domain.compute_Jn(alwaysUp, 1001)
    print(domain.J_N[1000])


    #########
    # SECTION 4: Optimal policy
    #########

    print("")
    print("Section 4:")
    print("")

    # Deterministic setting
    print("Deterministic domain:")
    print("")

    domain = Domain.DeterministicDomain(discount, rewardMatrix, (0, 0))
    domain.compute_Q(1001)
    policy = domain.retrieve_policy(1000)
    domain.compute_Jn(policy, 1001)
    print(policy.matrix)

    print("")
    print("Associated expected cumulative reward:")
    print("")

    print(domain.J_N[1000])

    print("")
    print("")

    # Stochastic setting
    print("Stochastic domain:")
    print("")

    domain = Domain.StochasticDomain(discount, 0.5, rewardMatrix, (0, 0))
    domain.compute_Q(1001)
    policy = domain.retrieve_policy(1000)
    domain.compute_Jn(policy, 1001)
    print(policy.matrix)

    print("")
    print("Associated expected cumulative reward:")
    print("")

    print(domain.J_N[1000])

    print("")
    print("")

    #########
    # SECTION 5: System identification
    #########

    print("")
    print("Section 5:")
    print("")

    moves = ["UP", "DOWN", "LEFT", "RIGHT"]

    print("Convergence toward r and p:")
    print("(t = 100000 and t = 1000000 are omitted to speed-up the execution; see full results in the report)")
    print("")

    step = [10, 100, 1000, 10000]
    for s in step:
        print("Trajectory of", s, "moves")
        domain = Domain.DeterministicDomain(discount, rewardMatrix, (3, 2))
        for t in range(0, s):
            # Randomly move on the grid
            index = np.random.randint(0, 4)
            direction = moves[index]
            domain.make_move(domain.pos, direction)

        # Estimate r and p based on the trajectory
        domain.estimate_r(domain.trajectory)
        domain.estimate_p(domain.trajectory)

        # Compute the estimated r values with the real values, using
        # the mean squared error
        SSEr = 0
        SSEp = 0
        for i in range(0, 5):
            for j in range(0, 5):
                k = 0
                for a in moves:
                    errorr = domain.compute_r(
                        (i, j), a) - (domain.estimated_R[i][j])[k]
                    for u in range(0, 5):
                        for v in range(0, 5):
                            errorp = domain.compute_p(
                                (u, v), (i, j), a) - ((domain.estimated_P[u][v])[i][j])[k]
                            SSEp += math.pow(errorp, 2)
                    SSEr += math.pow(errorr, 2)
                    k += 1
        MSEr = SSEr / 100
        MSEp = SSEp / 100
        print("MSE for r:", MSEr)
        print("MSE for p:", MSEp)

    print("")
    print("Comparison of the optimal policy found using exact and approximated values of Q_N")
    print("")

    # Deterministic setting

    domain = Domain.DeterministicDomain(discount, rewardMatrix, (3, 2))

    print("Deterministic domain:")
    print("")

    print("Exact Q_N:")
    print("")

    domain.compute_Q(1001)
    optimalPolicy = domain.retrieve_policy(1000)

    print(optimalPolicy.matrix)

    print("")
    print("Associated expected cumulative reward:")
    print("")

    domain.compute_Jn(optimalPolicy, 1001)
    print(domain.J_N[1000])
    print("")

    print("Approximated Q_N:")
    print("")

    # We first need to generate a trajectory
    for t in range(0, 100000):
        # Randomly move on the grid
        index = np.random.randint(0, 4)
        direction = moves[index]
        domain.make_move(domain.pos, direction)

    # We compute the corresponding optimal policy
    trajectory = domain.trajectory
    domain.estimate_Q(trajectory, 1001)
    optimalPolicy = domain.retrieve_policy(1000)

    print(optimalPolicy.matrix)

    print("")
    print("Associated expected cumulative reward:")
    print("")

    domain.compute_Jn(optimalPolicy, 1001)
    print(domain.J_N[1000])

    print("")
    print("")

    # Stochastic setting

    domain = Domain.StochasticDomain(discount, 0.1, rewardMatrix, (3, 2))

    print("Stochastic domain (beta = 0.1):")
    print("")

    print("Exact Q_N:")
    print("")

    domain.compute_Q(1001)
    optimalPolicy = domain.retrieve_policy(1000)

    print(optimalPolicy.matrix)

    print("")
    print("Associated expected cumulative reward:")
    print("")

    domain.compute_Jn(optimalPolicy, 1001)
    print(domain.J_N[1000])
    print("")

    print("Approximated Q_N:")
    print("")

    # We first need to generate a trajectory
    for t in range(0, 100000):
        # Randomly move on the grid
        index = np.random.randint(0, 4)
        direction = moves[index]
        domain.make_move(domain.pos, direction)

    # We compute the corresponding optimal policy
    trajectory = domain.trajectory
    domain.estimate_Q(trajectory, 1001)
    optimalPolicy = domain.retrieve_policy(1000)
    print(optimalPolicy.matrix)

    print("")
    print("Associated expected cumulative reward:")
    print("")

    domain.compute_Jn(optimalPolicy, 1001)
    print(domain.J_N[1000])

    print("")
    print("")
    """
    
    #########
    # SECTION 6: Q-Learning in a batch setting
    #########

    print("")
    print("Section 6:")
    print("")
    
    # Deterministic setting

    print("Deterministic domain:")
    print("")

    domain = Domain.DeterministicDomain(discount, rewardMatrix, (3, 3))

    print("Exact Q_N:")
    print("")

    domain.compute_Q(1001)
    optimalPolicy = domain.retrieve_policy(1000)

    print(optimalPolicy.matrix)

    print("")
    print("Associated expected cumulative reward:")
    print("")

    domain.compute_Jn(optimalPolicy, 1001)
    print(domain.J_N[1000])
    print("")

    print("Approximated Q:")
    print("")

    domain.Q_learning(10000, 0.05, 0.5, (3,3), True, batchSize=50)
    optimalPolicy = domain.retrieve_policy_Q_learning()
    print(optimalPolicy.matrix)

    print("")
    print("Associated expected cumulative reward:")
    print("")

    domain.compute_Jn(optimalPolicy, 1001)
    print(domain.J_N[1000])
    
    print("")
    print("Convergence toward Q:")
    print("(Fixed parameters: alpha=0.05, epsilon=0.5)")
    print("")

    domain = Domain.DeterministicDomain(discount, rewardMatrix, (3, 3))
    domain.compute_Q(1001)
    Q = domain.Q_N[1000]

    def test_convergence(domain, expRep=False, batchSize=100):
        moves = ["UP", "DOWN", "LEFT", "RIGHT"]
        step = [100, 1000, 2000, 5000, 10000]
        MAElist = []
        for s in step:    
            domain.Q_learning(s, 0.05, 0.5, (3,3), expRep, batchSize)
            
            SSE = 0
            for i in range(0, 5):
                for j in range(0, 5):
                    k = 0
                    for a in moves:
                        SSE += abs(((Q[i][j])[k] - (domain.Q[i][j])[k]))
                        k += 1
            MAE = SSE / 100
            MAElist.append(MAE)
        return MAElist

    def test_convergence_multi(domain, expRep=False, batchSize=100):
        moves = ["UP", "DOWN", "LEFT", "RIGHT"]
        step = [100, 1000, 2000, 5000, 10000]
        MAElist = []
        for s in step:
            for t in range(0,10):
                domain.Q_learning(s, 0.05, 0.5, (3,3), expRep, batchSize)
        
            SSE = 0
            for i in range(0, 5):
                for j in range(0, 5):
                    k = 0
                    for a in moves:
                        SSE += abs(((Q[i][j])[k] - (domain.Q[i][j])[k]))
                        k += 1
            MAE = SSE / 100
            MAElist.append(MAE)
        return MAElist

    domain = Domain.DeterministicDomain(discount, rewardMatrix, (3, 3))
    MAElistDet = test_convergence(domain)
    domain = Domain.DeterministicDomain(discount, rewardMatrix, (3, 3))
    MAElistDetMult = test_convergence_multi(domain)
    domain = Domain.DeterministicDomain(discount, rewardMatrix, (3, 3))
    MAElistDetRep = test_convergence(domain, expRep=True, batchSize=500)

    domain = Domain.StochasticDomain(discount, 0.20, rewardMatrix, (3, 3))
    MAElistSto = test_convergence(domain)
    domain = Domain.StochasticDomain(discount, 0.20, rewardMatrix, (3, 3))
    MAElistStoMult = test_convergence_multi(domain)
    domain = Domain.StochasticDomain(discount, 0.20, rewardMatrix, (3, 3))
    MAElistStoRep = test_convergence(domain, expRep=True, batchSize=500)

    x = [1, 2, 3, 4, 5]
    fig = plt.figure()
    ax = fig.add_subplot(111)
    plt.title("Convergence toward Q: MAE")
    ax.set_xlabel("Number of moves in the trajectory")
    ax.set_ylabel("Mean absolute error")
    plt.xticks(np.arange(6), ('0', '100', '1000', '2000', '5000', '10000'))
        
    plt.plot(x, MAElistDet, label="Deterministic, 1 traj.")
    plt.plot(x, MAElistDetMult, label="Deterministic, 10 traj.")
    plt.plot(x, MAElistDetRep, label="Deterministic, exp. replay")

    plt.plot(x, MAElistSto, label="Stochastic, 1 traj.")
    plt.plot(x, MAElistStoMult, label="Stochastic, 10 traj.")
    plt.plot(x, MAElistStoRep, label="Stochastic, exp. replay")
    plt.legend()

    plt.savefig('convergence.pdf')
    plt.show()
    plt.close()

    print("")
    print("Test of the intelligent agent:")
    print("")

    domain = Domain.DeterministicDomain(discount, rewardMatrix, (3, 3))

    print("Exact J_N:")
    print("")

    domain.compute_Q(1001)
    optimalPolicy = domain.retrieve_policy(1000)
    domain.compute_Jn(optimalPolicy, 1001)
    print(domain.J_N[1000])

    print("")
    print("Approximated Q and J_N:")
    print("")

    print("Epsilon = 0.05:")
    domain = Domain.DeterministicDomain(discount, rewardMatrix, (3, 3))
    for i in range(0,100):
        domain.Q_learning(1000, 0.05, 0.05, (3,3), False, batchSize=50)
    optimalPolicy = domain.retrieve_policy_Q_learning()
    print(optimalPolicy.matrix)
    domain.compute_Jn(optimalPolicy, 1001)
    print(domain.J_N[1000])
        
    print("Epsilon = 0.2:")
    domain = Domain.DeterministicDomain(discount, rewardMatrix, (3, 3))
    for i in range(0,100):
        domain.Q_learning(1000, 0.05, 0.2, (3,3), False, batchSize=50)
    optimalPolicy = domain.retrieve_policy_Q_learning()
    print(optimalPolicy.matrix)
    domain.compute_Jn(optimalPolicy, 1001)
    print(domain.J_N[1000])

    print("Epsilon = 0.5:")
    domain = Domain.DeterministicDomain(discount, rewardMatrix, (3, 3))
    for i in range(0,100):
        domain.Q_learning(1000, 0.05, 0.5, (3,3), False, batchSize=50)
    optimalPolicy = domain.retrieve_policy_Q_learning()
    print(optimalPolicy.matrix)
    domain.compute_Jn(optimalPolicy, 1001)
    print(domain.J_N[1000])


do_assignement(0.99)

# To redo the assignment with a discount factor of 0.4
#do_assignement(0.4)