import numpy as np
import matplotlib.pyplot as plt
import math
import display_caronthehill
import Domain
import Policy

discount = 0.95

#########
# SECTION 2: Implementation of the domain
#########

domain = Domain.Domain(discount)
for t in range(0, 200):
    # print(domain.state)
    ret = domain.make_move(4)
    if(ret):
        break
    # Line used to generate game pictures to produce a movie
    #display_caronthehill.save_caronthehill_image(domain.state[0], domain.state[1], "test{}.jpg".format(t))

#########
# SECTION 3: Expected return of a policy
#########

print("")
print("Section 3:")

print("Expected return of the always-choose-4 policy:")
# Creation of the always-choose-4 policy (cf. assignment statement)
policyMatrix = np.zeros((9, 9), dtype=object)
for i in range(0, 9):
    for j in range(0, 9):
        policyMatrix[i][j] = 4
always4 = Policy.Policy(policyMatrix)

print("")
print("Matrix form:")
print("")
domain = Domain.Domain(discount)
domain.compute_Jn(always4, 10)
print(domain.J_N)

print("")
print("Corresponding plot")
# Plot to better visualize the expected return values
fig = plt.matshow(domain.J_N)
plt.xlabel("Initial speed")
plt.ylabel("Initial position")
plt.xticks(
    np.arange(9),
    ('-2.67',
     '-2',
     '-1.33',
     '-0.67',
     '0',
     '0.67',
     '1.33',
     '2',
     '2.67'))
plt.yticks(
    np.arange(9),
    ('-0.88',
     '-0.66',
     '-0.44',
     '-0.22',
     '0',
     '0.22',
     '0.44',
     '0.66',
     '0.88'))
plt.colorbar()
plt.savefig('expected.pdf')
plt.show()
plt.close()

#########
# SECTION 5: Fitted-Q-Iteration
#########

print("")
print("Section 5:")

# First, we have to generate a (random) trajectory
domain = Domain.Domain(discount)
    
domain.fitted_Q_iteration(1, 5)
optimalPolicy = Policy.Policy(policyMatrix=None, model=domain.Q_N[len(domain.Q_N)-1], useModel=True)
domain.compute_Jn(optimalPolicy, 5)
print(domain.J_N)
fig = plt.matshow(domain.J_N)

