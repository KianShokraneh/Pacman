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

from learningAgents import ValueEstimationAgent
import collections


class ValueIterationAgent(ValueEstimationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A ValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=100):
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
        self.iterations = iterations
        self.values = util.Counter()  # A Counter is a dict with default 0
        self.runValueIteration()

    def runValueIteration(self):
        # Write value iteration code here
        for k in range(self.iterations):
            updated_values = util.Counter()
            for state in self.mdp.getStates():
                if self.mdp.isTerminal(state):
                    continue

                max_q = -999999999999999
                for action in self.mdp.getPossibleActions(state):
                    curr_q = self.computeQValueFromValues(state, action)
                    if curr_q > max_q:
                        max_q = curr_q
                updated_values[state] = max_q

            self.values = updated_values

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
        q = 0
        for t in self.mdp.getTransitionStatesAndProbs(state, action):
            successor_state, probability = t
            reward = self.mdp.getReward(state, action, successor_state)
            q += probability * (reward + self.discount * self.values[successor_state])
        return q

    def computeActionFromValues(self, state):
        """
          The policy is the best action in the given state
          according to the values currently stored in self.values.

          You may break ties any way you see fit.  Note that if
          there are no legal actions, which is the case at the
          terminal state, you should return None.
        """
        if self.mdp.isTerminal(state):
            return None

        max_q = -99999999999999
        best_policy = None
        for action in self.mdp.getPossibleActions(state):
            curr_q = self.computeQValueFromValues(state, action)
            if curr_q > max_q:
                max_q = curr_q
                best_policy = action

        return best_policy

    def getPolicy(self, state):
        return self.computeActionFromValues(state)

    def getAction(self, state):
        "Returns the policy at the state (no exploration)."
        return self.computeActionFromValues(state)

    def getQValue(self, state, action):
        return self.computeQValueFromValues(state, action)


class AsynchronousValueIterationAgent(ValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        An AsynchronousValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs cyclic value iteration
        for a given number of iterations using the supplied
        discount factor.
    """

    def __init__(self, mdp, discount=0.9, iterations=1000):
        """
          Your cyclic value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy. Each iteration
          updates the value of only one state, which cycles through
          the states list. If the chosen state is terminal, nothing
          happens in that iteration.

          Some useful mdp methods you will use:
              mdp.getStates()
              mdp.getPossibleActions(state)
              mdp.getTransitionStatesAndProbs(state, action)
              mdp.getReward(state)
              mdp.isTerminal(state)
        """
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        for k in range(self.iterations):
            states = self.mdp.getStates()
            curr_state = states[k % len(states)]
            if self.mdp.isTerminal(curr_state):
                continue

            max_q = -999999999999999
            for action in self.mdp.getPossibleActions(curr_state):
                curr_q = self.computeQValueFromValues(curr_state, action)
                if curr_q > max_q:
                    max_q = curr_q
            self.values[curr_state] = max_q


class PrioritizedSweepingValueIterationAgent(AsynchronousValueIterationAgent):
    """
        * Please read learningAgents.py before reading this.*

        A PrioritizedSweepingValueIterationAgent takes a Markov decision process
        (see mdp.py) on initialization and runs prioritized sweeping value iteration
        for a given number of iterations using the supplied parameters.
    """

    def __init__(self, mdp, discount=0.9, iterations=100, theta=1e-5):
        """
          Your prioritized sweeping value iteration agent should take an mdp on
          construction, run the indicated number of iterations,
          and then act according to the resulting policy.
        """
        self.theta = theta
        ValueIterationAgent.__init__(self, mdp, discount, iterations)

    def runValueIteration(self):
        pq = util.PriorityQueue()
        predecessors = {s: set() for s in self.mdp.getStates()}

        for s in self.mdp.getStates():
            if self.mdp.isTerminal(s):
                continue

            max_q = -9999999999999
            for action in self.mdp.getPossibleActions(s):
                q = self.computeQValueFromValues(s, action)
                if q > max_q:
                    max_q = q

                for t in self.mdp.getTransitionStatesAndProbs(s, action):
                    if t[1]:
                        predecessors[t[0]].add(s)

            pq.push(s, -abs(max_q - self.values[s]))

        for i in range(self.iterations):
            if pq.isEmpty():
                break
            s = pq.pop()
            if self.mdp.isTerminal(s):
                continue
            actions = self.mdp.getPossibleActions(s)
            self.values[s] = max([self.computeQValueFromValues(s, action) for action in actions])

            for pre in predecessors[s]:
                actions = self.mdp.getPossibleActions(pre)
                q = [self.computeQValueFromValues(pre, action) for action in actions]
                difference = abs(self.values[pre] - max(q))
                if difference > self.theta:
                    pq.update(pre, -difference)
