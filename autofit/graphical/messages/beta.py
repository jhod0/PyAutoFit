
import numpy as np
from scipy.special import betaln, psi

from ..utils import cached_property, inv_beta_suffstats
from ..messages.abstract import AbstractMessage



class BetaMessage(AbstractMessage):
    """
    Models a Beta distribution
    """
    log_base_measure = 0
    _support = ((0, 1),)
    _min = 0
    _max = 1
    _range = 1
    _parameter_support = ((0, np.inf), (0, np.inf))
    
    def __init__(self, alpha=0.5, beta=0.5, log_norm=0):
        self.alpha = alpha
        self.beta = beta           
        super().__init__(
            alpha, beta, log_norm=log_norm
        )
    
    @cached_property
    def log_partition(self) -> np.ndarray:
        return betaln(*self.parameters)
    
    @cached_property
    def natural_parameters(self) -> np.ndarray:
        return self.calc_natural_parameters(
            self.alpha,
            self.beta
        )
    
    @staticmethod
    def calc_natural_parameters(
            alpha: np.ndarray, beta: np.ndarray
    ) -> np.ndarray:
        return np.array([alpha - 1, beta - 1]) 
    
    @staticmethod
    def invert_natural_parameters(
            natural_parameters: np.ndarray
    ) -> np.ndarray:
        return natural_parameters + 1
    
    @classmethod
    def invert_sufficient_statistics(
            cls, sufficient_statistics: np.ndarray
    ) -> np.ndarray:
        a, b = inv_beta_suffstats(*sufficient_statistics)
        return cls.calc_natural_parameters(a, b)
    
    @classmethod
    def to_canonical_form(cls, x: np.ndarray) -> np.ndarray:
        return np.array([np.log(x), np.log1p(-x)])

    @cached_property
    def mean(self) -> np.ndarray:
        return self.alpha / (self.alpha + self.beta)

    @cached_property
    def variance(self) -> np.ndarray:
        return (
            self.alpha * self.beta
            / (self.alpha + self.beta)**2 
            / (self.alpha + self.beta + 1)
        )

    def sample(self, n_samples=None):
        a, b = self.parameters
        shape = (n_samples,) + self.shape if n_samples else self.shape
        return np.random.beta(a, b, size=shape)

    def kl(self, dist):
        # TODO check this is correct
        # https://arxiv.org/pdf/0911.4863.pdf
        if self._support != dist._support:
            raise TypeError('Support does not match')
            
        P, Q = self, dist
        aP, bP = dist.parameters
        aQ, bQ = self.parameters
        return (
            betaln(aQ, bQ) - betaln(aP, bP)
            - (aQ - aP) * psi(aP)
            - (bQ - bP) * psi(bP)
            + (aQ - aP + bQ - bP) * psi(aP + bP)
        )

    def logpdf_gradient(self, x):
        logl = self.logpdf(x)
        a, b = self.parameters
        gradl = (a-1)/x + (b-1)/(x-1)
        return logl, gradl 

    def logpdf_gradient_hessian(self, x):
        logl = self.logpdf(x)
        a, b = self.parameters
        ax, bx = (a-1)/x, (b-1)/(x-1)
        gradl = ax + bx
        hessl = -ax/x - bx/(x-1)
        return logl, gradl, hessl