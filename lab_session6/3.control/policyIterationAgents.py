# valueIterationAgents.py
# -----------------------
# Licensing Information:  You are free to use or extend these projects for
# educational purposes provided that (1) you do not distribute or publish
# solutions, (2) you retain this notice, and (3) you provide clear
# attribution to UC Berkeley, including a link to http://ai.berkeley.edu.
#
# Attribution Information: The Pacman AI projects were developed at UC Berkeley.
# The core projects and autograders were primarily created by John DeNero
# (denero@cs.berkeley.edu) and Dan Klein (klein@cs.berkeley.edu).
# Student side autograding was added by Brad Miller, Nick Hay, and
# Pieter Abbeel (pabbeel@cs.berkeley.edu).


import mdp, util
import numpy as np
from learningAgents import ValueEstimationAgent

class PolicyIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PolicyIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs policy iteration
        for a given number of iterations using the supplied
        discount factor.
    """
    def __init__(self, mdp, discount = 0.9, iterations = 100):
        """
          Your value iteration agent should take an mdp on
          construction, run the indicated number of iterations
          and then act according to the resulting policy.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state, action, nextState)
              mdp.isTerminal(state)
        """
        self.mdp = mdp
        self.discount = discount
        print("using discount {}".format(discount))
        self.iterations = iterations
        self.values = util.Counter() # A Counter is a dict with default 0
        self.policies = util.Counter() # A Counter is a dict with default 0
        delta = 0.01
        # TODO: Implement Policy Iteration.
        # Exit either when the number of iterations is reached,
        # OR until convergence (L2 distance < delta).
        # Print the number of iterations to convergence.
        # To make the comparison FAIR, one iteration is a single sweep over states.
        # Compute the number of steps until policy convergence, but do not stop
        # the algorithm until values converge. #TODO

        # Init values
        for s in mdp.getStates():
            self.values[s] = 0
            if mdp.isTerminal(s):
                continue
            self.policies[s] = mdp.getPossibleActions(s)[0]

        state_iters = 0    # Iterations over state space until policy convergerce
        policy_iters = 0   # Iterations over algorithm until policy convergerce
        algo_iters = 0

        def L2_norm(v1, v2):
            dist = 0
            for k in v1.keys():
                dist += (v1[k] - v2[k]) ** 2
            return dist**(1/2)

        policy_stable = False
        values_converged = False

        while not values_converged and algo_iters != iterations:
            # Policy Evaluation
            dist = delta
            while dist >= delta:
                old_values = self.values.copy()
                for s in mdp.getStates():
                    # Skip terminal state
                    if mdp.isTerminal(s):
                        continue
                    v = self.values[s]
                    new_v = 0

                    for s_n, p in mdp.getTransitionStatesAndProbs(s, self.policies[s]):
                        new_v += p * (mdp.getReward(s, self.policies[s], s_n) + discount*self.values[s_n])
                    self.values[s] = new_v

                # Calculate the new distance
                dist = L2_norm(self.values, old_values)
                if not policy_stable:
                    state_iters += 1
            values_converged = True

            # Policy Improvement
            if not policy_stable:
                policy_iters += 1
                state_iters += 1

            policy_stable = True
            for s in mdp.getStates():
                if mdp.isTerminal(s):
                    continue

                old_action = self.policies[s]

                p_list = list()
                possible_actions = mdp.getPossibleActions(s)
                for a in possible_actions:
                    v_sum = 0
                    for s_n, p in mdp.getTransitionStatesAndProbs(s, a):
                        v_sum += p * (mdp.getReward(s, a, s_n) + discount * self.values[s_n])
                    p_list.append(v_sum)
                # Assign the maximum value to the current state
                self.policies[s] = possible_actions[np.argmax(p_list)]

                if old_action != self.policies[s]:
                    policy_stable = False
                    values_converged = False

            algo_iters += 1

        print(f"Policy Iteration: {state_iters} iterations over the state space")
        print(f"Policy Iteration: {policy_iters} iterations until policy convergence")

    def getValue(self, state):
        """
          Return the value of the state (computed in __init__).
        """
        return self.values[state]


    def computeQValueFromValues(self, state, action):
        """
          Compute the Q-value of action in state from the
          value function stored in self.values.
        """
        # TODO: Implement this function according to the doc
        util.raiseNotDefined()


    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        # TODO: Implement according to the doc
        util.raiseNotDefined()

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)
