"""
Module containing the agent classes to solve a Bandit problem.

Complete the code wherever TODO is written.
Do not forget the documentation for every class and method!
An example can be seen on the Bandit_Agent and Random_Agent classes.
"""
# -*- coding: utf-8 -*-
import numpy as np
from utils import softmax, my_random_choice


class Bandit_Agent(object):
    """
    Abstract Agent to solve a Bandit problem.

    Contains the methods learn() and act() for the base life cycle of an agent.
    The reset() method reinitializes the agent.
    The minimum requirement to instantiate a child class of Bandit_Agent
    is that it implements the act() method (see Random_Agent).
    """

    def __init__(self, k: int, **kwargs):
        """
        Simply stores the number of arms of the Bandit problem.
        The __init__() method handles hyperparameters.
        Parameters
        ----------
        k: positive int
            Number of arms of the Bandit problem.
        kwargs: dictionary
            Additional parameters, ignored.
        """
        self.k = k

    def reset(self):
        """
        Reinitializes the agent to 0 knowledge, good as new.

        No inputs or outputs.
        The reset() method handles variables.
        """
        pass

    def learn(self, a: int, r: float):
        """
        Learning method. The agent learns that action a yielded reward r.
        Parameters
        ----------
        a: positive int < k
            Action that yielded the received reward r.
        r: float
            Reward for having performed action a.
        """
        pass

    def act(self) -> int:
        """
        Agent's method to select a lever (or Bandit) to pull.
        Returns
        -------
        a : positive int < k
            The action the agent chose to perform.
        """
        raise NotImplementedError("Calling method act() in Abstract class Bandit_Agent")


class Random_Agent(Bandit_Agent):
    """
    This agent doesn't learn, just acts purely randomly.
    Good baseline to compare to other agents.
    """

    def act(self):
        """
        Random action selection.
        Returns
        -------
        a : positive int < k
            A randomly selected action.
        """
        return np.random.randint(self.k)


class EpsGreedy_SampleAverage(Bandit_Agent):
    # This class uses Sample Averages to estimate q; others are non stationary.
    def __init__(self, k: int, eps: float, **kwargs):
        """
        Stores the number of arms of the Bandit problem and the probability with which to select a random action.
        The __init__() method handles hyperparameters.
        Parameters
        ----------
        k: positive int
            Number of arms of the Bandit problem.
        eps: positive float
            Probability with which to select a random action
        kwargs: dictionary
            Additional parameters, ignored.
        """
        self.k = k
        self.eps = eps

        self.means = list()
        for _ in range(self.k):
            self.means.append([0.0, 0])

    def reset(self):
        """
        Reinitializes the agent to 0 knowledge, good as new.

        No inputs or outputs.
        The reset() method handles variables.
        """
        for i in range(self.k):
            self.means[i] = [0.0, 0]

    def learn(self, a: int, r: float):
        """
        Learning method. The agent learns that action a yielded reward r.
        Parameters
        ----------
        a: positive int < k
            Action that yielded the received reward r.
        r: float
            Reward for having performed action a.
        """
        Qn, n = self.means[a]
        Q_n1 = Qn + 1 / (n + 1) * (r - Qn)
        self.means[a] = [Q_n1, n + 1]

    def act(self) -> int:
        """
        Agent's method to select a lever (or Bandit) to pull.
        Selects a random action with probability eps.
        Returns
        -------
        a : positive int < k
            The action the agent chose to perform.
        """
        n = np.random.random_sample()
        if n <= self.eps:
            return np.random.randint(self.k)
        else:
            return np.argmax([Qn for Qn, n in self.means])


class EpsGreedy(Bandit_Agent):
    # Non stationary agent with q estimating and eps-greedy action selection.
    def __init__(self, k: int, eps: float, alpha: float, **kwargs):
        """
        Stores the number of arms of the Bandit problem and the probability with which to select a random action.
        The __init__() method handles hyperparameters.
        Parameters
        ----------
        k: positive int
            Number of arms of the Bandit problem.
        eps: positive float
            Probability with which to select a random action
        kwargs: dictionary
            Additional parameters, ignored.
        """
        self.k = k
        self.eps = eps
        self.alpha = alpha

        self.means = [0.0] * self.k

    def reset(self):
        """
        Reinitializes the agent to 0 knowledge, good as new.

        No inputs or outputs.
        The reset() method handles variables.
        """
        for i in range(self.k):
            self.means[i] = 0.0

    def learn(self, a: int, r: float):
        """
        Learning method. The agent learns that action a yielded reward r.
        Parameters
        ----------
        a: positive int < k
            Action that yielded the received reward r.
        r: float
            Reward for having performed action a.
        """
        Qn = self.means[a]
        Q_n1 = Qn + self.alpha * (r - Qn)
        self.means[a] = Q_n1

    def act(self) -> int:
        """
        Agent's method to select a lever (or Bandit) to pull.
        Selects a random action with probability eps.
        Returns
        -------
        a : positive int < k
            The action the agent chose to perform.
        """
        n = np.random.random_sample()
        if n <= self.eps:
            return np.random.randint(self.k)
        else:
            return np.argmax([Qn for Qn in self.means])


class OptimisticGreedy(EpsGreedy):
    # Same as above but with optimistic starting values.
    def __init__(self, q0: float, **kwargs):
        """
        Simply stores the number of arms of the Bandit problem.
        The __init__() method handles hyperparameters.
        Parameters
        ----------
        q0: float
            The optimistic starting value
        kwargs: dictionary
            Additional parameters, ignored.
        """
        super().__init__(**kwargs)
        self.q0 = q0
        self.reset()

    def reset(self):
        """
        Reinitializes the agent to 0 knowledge, good as new.

        No inputs or outputs.
        The reset() method handles variables.
        """
        for i in range(self.k):
            self.means[i] = self.q0


class UCB(Bandit_Agent):
    def __init__(self, k: int, c: float, **kwargs):
        """
        Stores the number of arms of the Bandit problem.
        The __init__() method handles hyperparameters.
        Parameters
        ----------
        k: positive int
            Number of arms of the Bandit problem.
        c: positive float
            Weight of the action's potential
        kwargs: dictionary
            Additional parameters, ignored.
        """
        self.k = k
        self.c = c

        self.t = 1
        self.means = [0.0] * self.k

        self.means = list()
        for _ in range(self.k):
            self.means.append([0.0, 0])

    def reset(self):
        """
        Reinitializes the agent to 0 knowledge, good as new.

        No inputs or outputs.
        The reset() method handles variables.
        """
        self.t = 0
        for i in range(self.k):
            self.means[i] = [0.0, 0]

    def learn(self, a: int, r: float):
        """
        Learning method. The agent learns that action a yielded reward r.
        Parameters
        ----------
        a: positive int < k
            Action that yielded the received reward r.
        r: float
            Reward for having performed action a.
        """
        self.t += 1

        Qn, n = self.means[a]
        Q_n1 = Qn + 1 / (n + 1) * (r - Qn)
        self.means[a] = [Q_n1, n + 1]

    def act(self) -> int:
        """
        Agent's method to select a lever (or Bandit) to pull.
        Returns
        -------
        a : positive int < k
            The action the agent chose to perform.
        """

        if self.t < self.k:
            action = self.t
        else:
            action = np.argmax([Qt + self.c * np.sqrt(np.log(self.t) / n) for Qt, n in self.means])

        return action


class Gradient_Bandit(Bandit_Agent):
    # If you want this to run fast, use the my_random_choice function from
    # utils instead of np.random.choice to sample from the softmax
    # You can also find the softmax function in utils.

    def __init__(self, k: int, alpha: float, **kwargs):
        """
        Simply stores the number of arms of the Bandit problem.
        The __init__() method handles hyperparameters.
        Parameters
        ----------
        k: positive int
            Number of arms of the Bandit problem.
        eps: positive float
            Probability with which to select a random action
        kwargs: dictionary
            Additional parameters, ignored.
        """
        self.k = k
        self.alpha = alpha

        self.reset()

    def reset(self):
        """
        Reinitializes the agent to 0 knowledge, good as new.

        No inputs or outputs.
        The reset() method handles variables.
        """
        self.t = 0
        self.mean = 0

        self.H = np.zeros(self.k)

    def learn(self, a: int, r: float):
        """
        Learning method. The agent learns that action a yielded reward r.
        Parameters
        ----------
        a: positive int < k
            Action that yielded the received reward r.
        r: float
            Reward for having performed action a.
        """
        self.t += 1
        self.mean += (1 / self.t) * (r - self.mean)

        softm = softmax(self.H)
        for i in range(self.k):
            if i == a:
                self.H[i] += self.alpha * (r - self.mean) * (1 - softm[i])
            else:
                self.H[i] -= self.alpha * (r - self.mean) * softm[i]

    def act(self) -> int:
        """
        Agent's method to select a lever (or Bandit) to pull.
        Returns
        -------
        a : positive int < k
            The action the agent chose to perform.
        """
        return my_random_choice(self.k, softmax(self.H))
