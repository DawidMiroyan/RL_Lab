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


class Gaussian_Bandit(Bandit):
    # Reminder: the Gaussian_Bandit's distribution is a fixed Gaussian.
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs: dictionary
            Ignored additional inputs.
        """
        super(Gaussian_Bandit, self).__init__(**kwargs)

        self.mean = 0  # Mean
        self.reset()

    def reset(self):
        """
        Reinitializes the distribution.
        """
        self.mean = np.random.normal(0, 1)

    def pull(self) -> float:
        """
        Returns a sample from the distribution.
        """
        return np.random.normal(self.mean, 1)


class Gaussian_Bandit_NonStat(Bandit):
    # Reminder: the distribution mean changes each step over time,
    # with increments following N(m=0,std=0.01)
    def __init__(self, **kwargs):
        """
        Parameters
        ----------
        **kwargs: dictionary
            Ignored additional inputs.
        """
        super(Gaussian_Bandit_NonStat, self).__init__(**kwargs)

        self.mean = 0  # Mean
        self.reset()

    def reset(self):
        """
        Reinitializes the distribution.
        """
        self.mean = np.random.normal(0, 1)

    def pull(self) -> float:
        """
        Returns a sample from the distribution.
        """
        return np.random.normal(self.mean, 1)

    def update_mean(self):
        self.mean += np.random.normal(0, 0.01)


class KBandit(Bandit):
    """ Set of k Gaussian_Bandits. """

    def __init__(self, k, **kwargs):
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
        self.bandits = [Gaussian_Bandit() for _ in range(self.k)]

    def reset(self):
        """ Resets each of the k bandits. """
        for bandit in self.bandits:
            bandit.reset()
        self.best_action = np.argmax([bandit.mean for bandit in self.bandits])  # printing purposes

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
        return self.bandits[action].pull()

    def is_best_action(self, action: int) -> int:
        """
        Checks if pulling from Bandit using #action is the best action.
        """
        return int(action == self.best_action)


class KBandit_NonStat(Bandit):
    # Reminder: Same as KBandit, with non stationary Bandits.
    """ Set of k Gasussian_Bandits_NonStat """

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
        self.bandits = [Gaussian_Bandit_NonStat() for _ in range(self.k)]

    def reset(self):
        """ Resets each of the k bandits """
        for bandit in self.bandits:
            bandit.reset()
        self.best_action = np.argmax([bandit.mean for bandit in self.bandits])  # printing purposes

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
        self.best_action = np.argmax([bandit.mean for bandit in self.bandits])  # printing purposes

    def is_best_action(self, action: int) -> int:
        """
        Checks if pulling from Bandit using #action is the best action.
        """
        return int(action == self.best_action)
