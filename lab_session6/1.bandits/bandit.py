"""
Module containing the k-armed bandit problem
Complete the code wherever TODO is written.
Do not forget the documentation for every class and method!
We expect all classes to follow the Bandit abstract object formalism.
"""
# -*- coding: utf-8 -*-
import numpy as np


class Bandit(object):
    """
    Abstract concept of a Bandit, i.e. Slot Machine, the Agent can pull.

    A Bandit is a distribution over reals.
    The pull() method samples from the distribution to give out a reward.
    """

    def __init__(self, **kwargs):
        """
        Empty for our simple one-armed bandits, without hyperparameters.
        Parameters
        ----------
        **kwargs: dictionary
            Ignored additional inputs.
        """
        pass

    def reset(self):
        """
        Reinitializes the distribution.
        """
        pass

    def pull(self) -> float:
        """
        Returns a sample from the distribution.
        """
        raise NotImplementedError("Calling method pull() in Abstract class Bandit")


class Mixture_Bandit_NonStat(Bandit):
    """ A Mixture_Bandit_NonStat is a 2-component Gaussian Mixture
    reward distribution (sum of two Gaussians with weights w and 1-w in [O,1]).

    The two means are selected according to N(0,1) as before.
    The two weights of the gaussian mixture are selected uniformly.
    The Gaussian mixture is non-stationary: the means AND WEIGHTS move every
    time-step by an increment epsilon~N(m=0,std=0.01)"""

    # TODO: Implement this class inheriting the Bandit above.
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs: dictionary
            Ignored additional inputs.
        """
        self.mean1 = 0.0  # Mean 1
        self.mean2 = 0.0  # Mean 2

        self.weight1 = 1.0  # Weight 1
        self.weight2 = 0.0  # Weight 2

        self.reset()

    def reset(self):
        """
        Reinitializes the distribution.
        """
        self.mean1 = np.random.normal(0, 1)
        self.mean2 = np.random.normal(0, 1)

        self.weight1 = np.random.uniform()
        self.weight2 = 1.0 - self.weight1

    def pull(self) -> float:
        """
        Returns a sample from the distribution.
        """
        return np.random.normal(self.mean1, 1) * self.weight1 + \
               np.random.normal(self.mean2, 1) * self.weight2

    def update_mean(self):
        self.mean1 += np.random.normal(0, 0.01)
        self.mean2 += np.random.normal(0, 0.01)
        self.weight1 += np.random.normal(0, 0.01)
        self.weight2 = 1.0 - self.weight1


class KBandit_NonStat:
    """ Set of K Mixture_Bandit_NonStat Bandits.
    The Bandits are non stationary, i.e. every pull changes all the
    distributions.

    This k-armed Bandit has:
    * an __init__ method to initialize k
    * a reset() method to reset all Bandits
    * a pull(lever) method to pull one of the Bandits; + non stationarity
    """

    def __init__(self, k, **config):
        """
        Instantiates the k-armed bandit, with a number of arms, and initializes
        the set of bandits to new gaussian bandits in a bandits list.
        The reset() method is supposedly called from outside.
        Parameters
        ----------
        k: positive int
            Number of arms of the problem.
        """
        self.k = k
        self.best_action = 0
        self.bandits = [Mixture_Bandit_NonStat() for _ in range(self.k)]

    def reset(self):
        """ Resets each of the k bandits """
        for bandit in self.bandits:
            bandit.reset()
        self.best_action = np.argmax([bandit.mean1 * bandit.weight1 + bandit.mean2 * bandit.weight2 for bandit in self.bandits])  # printing purposes

    def pull(self, action: int) -> float:
        """
        Pulls the lever from Bandit #action. Returns the reward.
        Parameters
        ----------
        action: positive int < k
            Lever to pull.
        Returns
        -------
        reward : float
            Reward for pulling this lever.
        """
        r = self.bandits[action].pull()
        self.update_means()
        return r

    def update_means(self):
        """ Updates the mean for each of the Bandits"""
        for bandit in self.bandits:
            bandit.update_mean()
        self.best_action = np.argmax([bandit.mean1 * bandit.weight1 + bandit.mean2 * bandit.weight2 for bandit in self.bandits])

    def is_best_action(self, action: int) -> int:
        """
        Checks if pulling from Bandit using #action is the best action.
        """
        return int(action == self.best_action)
