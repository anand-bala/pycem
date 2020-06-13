"""Provides the CEM class, the core of this package"""

import math
import logging
from typing import Union, Mapping, Callable, Optional, Sequence

import attr
import numpy as np
from scipy.stats import truncnorm


@attr.s(slots=True, auto_attribs=True, order=False)
class CEMResults:
    """Data class containing the results of the optimization procedure, with some other metadata"""

    xbest: np.ndarray = attr.ib()
    fbest: np.ndarray = attr.ib()
    iterations: int = attr.ib()
    stop: "CEMStopCondition" = attr.ib()


@attr.s(slots=True, auto_attribs=True, order=False)
class CEMStopCondition:
    """Information regarding whether the optimization has ended and, if so, why"""

    stopped: bool = attr.ib(default=False)
    max_iters: int = attr.ib(default=-1)
    epsilon: float = attr.ib(default=-math.inf)

    def __bool__(self):
        return self.stopped


class CEMOptimizer:
    """CEM Optimizer class.

    Performs CEM optimization for a continuous domain, with Gaussian sampling/updates.

    Parameters
    ----------
    x0 : np.ndarray
        Initial mean for the CEM Gaussian distribution.
    sigma0 : np.ndarray
        Initial stddev for CEM Gaussian distribution.
    max_iters : int
        Maximum number of optimization iterations.
    popsize : int
        Number of solutions to sample during each optimizer iteration.
    num_elites : int
        Number of "elites" to pick during each optimization iteration.
    alpha: float
        Smoothing hyperparameter (default = 0.25).
    epsilon: float
        Minimum variance such that if the maximum variance drops below `epsilon` the optimization is stopped (default = 0.001).
    upper_bound :
        The upper bound (as an array) for the normal distribution (optional).
    lower_bound :
        The lower bound (as an array) for the normal distribution (optional).
    rng : np.random.Generator
        A Numpy random generator to use for sampling, etc.
    """

    def __init__(
        self,
        x0: np.ndarray,
        sigma0: np.ndarray,
        *,
        max_iters: int,
        popsize: int,
        num_elites: int,
        alpha: float = 0.25,
        epsilon: float = 0.001,
        upper_bound: np.ndarray = None,
        lower_bound: np.ndarray = None,
        rng: np.random.Generator = None,
        random_seed: Union[
            int,
            Sequence[int],
            np.random.SeedSequence,
            np.random.BitGenerator,
            np.random.Generator,
        ] = None,
    ):
        self.logger = logging.getLogger(__name__)
        if rng is None:
            rng = np.random.default_rng(random_seed)
        elif isinstance(rng, np.random.Generator):
            raise ValueError(
                f"Not sure how to infer RNG of type {type(rng)} as a np.random.Generator"
            )
        self.rng = rng

        self.x0 = np.asarray(x0)
        self.sigma0 = np.asarray(sigma0)
        if self.x0.shape != self.sigma0.shape:
            raise ValueError(
                "This CEM optimizer is not smart enough to infer the size of your solution if your x0 and sigma0 have different sizes."
            )
        if np.any(self.sigma0 < 0):
            raise ValueError("Standard deviation value cannot be less than 0")
        if np.any(self.sigma0 == 0):
            self.logger.warning(
                "Some sigma0 value is set to 0. This means that we will be sampling from a point distribution for that parameter, and hence cannot optimize anything there."
            )

        self.max_iters = max_iters
        if self.max_iters <= 0:
            raise ValueError(
                f"[max_iters] It doesn't make sense to run the optimization for {max_iters} (<=0) steps."
            )
        self._popsize = popsize
        if self._popsize <= 0:
            raise ValueError(
                f"[popsize] It doesn't make sense to run the optimization with popsize {popsize} (<=0)."
            )

        self.num_elites = num_elites
        if self.num_elites <= 0:
            raise ValueError(f"[num_elites] Please provide a positive value...")
        self.alpha = alpha
        if not (0 <= self.alpha <= 1):
            raise ValueError(
                f"[alpha] Smoothing value alpha ({alpha}) has to be in range [0, 1]"
            )
        self.epsilon = epsilon
        if epsilon <= 0:
            raise ValueError(
                f"[epsilon] Setting the min variance to a non-positive value doesn't make sense..."
            )
        self.upper_bound = np.asarray(
            upper_bound if upper_bound is not None else math.inf
        )
        self.lower_bound = np.asarray(
            lower_bound if lower_bound is not None else -math.inf
        )
        if np.any(self.lower_bound >= self.upper_bound):
            raise ValueError("Lower bound >= Upper bound...")

        # Compute the variance, given the bounds.
        lb, ub = (
            np.abs(self.x0 - self.lower_bound),
            np.abs(self.upper_bound - self.x0),
        )
        self.constrained_sigma = np.minimum(np.minimum(lb / 2, ub / 2), self.sigma0)

        self.best_x = x0
        self.best_f = -math.inf
        self.t = 0

    @property
    def popsize(self) -> int:
        return self._popsize

    @property
    def best(self) -> CEMResults:
        return CEMResults(self.best_x, self.best_f, self.t, self.stop())

    @property
    def result(self) -> CEMResults:
        """Get the results for the CEM optimization"""
        return self.best

    def _is_bounded(self):
        if np.isinf(self.upper_bound).any() and np.isinf(self.lower_bound).any():
            return False
        if np.isfinite(self.upper_bound).all() and np.isfinite(self.lower_bound).all():
            return True
        # else one is bounded and the other is not, and I am not sure how to handle this...
        raise NotImplementedError(
            "I am not sure how to handle a (lb, ub) pair where only one is bounded..."
        )

    def ask(self, number: Optional[int] = None) -> np.ndarray:
        """Sample candidate solutions from the internal distribution

        Parameters
        ----------
        number : Optional[int]
            If given, it is used as the `number` of values to sample from the
            distribution. Else, `self.popsize` is used.

        Returns
        -------
        np.ndarray :
            A vector containing the desired number of samples. Shape is `(n_samples,) +
            x0.shape`.
        """
        popsize = number if number is not None else self.popsize
        out_size = (popsize,) + self.x0.shape
        self.logger.debug(f"Asked for {popsize} samples. Output size: {out_size}")

        if self._is_bounded():
            return truncnorm.rvs(
                -2,
                2,
                loc=self.x0,
                scale=self.constrained_sigma,
                size=out_size,
                random_state=self.rng,
            )
        return self.rng.normal(loc=self.x0, sigma0=self.sigma0, size=out_size)

    def tell(self, xs: np.ndarray, ys: np.ndarray):
        """Pass objective function values to prepare for next iteration. The optimizer
        naturally assumes we are maximizing things, so scale the objective accordingly

        Parameters
        ----------
        xs : np.ndarray
            Set of candidate inputs (probably generated by CEM). Expected shape
            `(popsize,) + x0.shape`
        ys : np.ndarray
            Set of fitness values for each candidate in the xs Tensor. Expected shape
            `(popsize,)`
        """
        self.logger.debug(
            f"Received evaluations for a set of samples of size {xs.shape}"
        )
        # First we rank the solutions
        indices = np.argpartition(ys, self.num_elites)
        # Maintain the current best solution
        self.best_x, self.best_f = xs[indices[0]], ys[indices[0]]
        # Get the new mean
        sigmat_ = np.std(xs[indices], axis=0)
        xt_ = np.mean(xs[indices], axis=0)
        # Smooth update
        self.x0 = self.alpha * xt_ + (1 - self.alpha) * self.x0
        self.sigma0 = self.alpha * sigmat_ + (1 - self.alpha) * self.sigma0

        self.t += 1

    def stop(self) -> CEMStopCondition:
        """Return the stopping criterion that has been met, or empty dict otherwise"""
        ret = CEMStopCondition(stopped=False, max_iters=-1, epsilon=-1.0)
        if self.t > self.max_iters:
            ret.max_iters = self.max_iters
            ret.stopped = True
        if self.sigma0.max() < self.epsilon:
            ret.epsilon = self.epsilon
            ret.stopped = True
        return ret

    def optimize(self, fitness_fn: Callable[[np.ndarray], np.ndarray]) -> "CEMResults":
        """Performs the optimization (maximization) loop given a fitness function.

        The fitness function is a callable that takes in a vector of shape
        `(popsize, x0.shape)` and outputs a vector of shape `(popsize,)`

        Parameters
        ----------
        fitness_fn : Callable[[np.ndarray], np.ndarray]
            The fitness function for a set of samples

        Returns
        -------
        CEMResults :
            Returns the results of the optimization.
        """
        while not self.stop():
            X = self.ask()
            y = np.asarray(fitness_fn(X))
            self.tell(X, y)
        return self.result
