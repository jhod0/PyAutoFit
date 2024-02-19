from typing import Optional
import numpy as np
from scipy import special as sp


class LogNormalPrior:
    __identifier_fields__ = ('lower_limit', 'upper_limit', 'mean', 'sigma_dex')
    __database_args__ = ('lower_limit', 'upper_limit', 'mean', 'sigma_dex', 'id_')

    def __init__(
        self,
        mean: float,
        sigma_dex: float,
        lower_limit: float = 0.0,
        upper_limit: float = float('inf'),
        id_: Optional[int] = None
    ):
        '''
        A prior describing a variable whose log is normally distributed.

        mean: The mean of the distribution in *real space*, meaning it should be a positive number.
        sigma_dex: The scatter of the distribution in *log space*, base 10
        lower_limit: Lower limit of the distribution. Optional.
        upper_limit: Upper limit of the distribution. Optional.
        '''
        self.lower_limit = float(lower_limit)
        self.upper_limit = float(upper_limit)

        # The mean in real space, and in natural log
        self.mean = mean
        self.log_mean = np.log(mean)

        # The scatter log 10, and natural log
        self.sigma_dex = sigma_dex
        self.sigma_ln = sigma_dex * np.log(10)

        self.log_upper_limit = np.log(self.upper_limit)

        if self.lower_limit <= 0.0:
            self.lower_limit = 0.0
            self._erf_lower = -1.0
        else:
            log_lower_limit = np.log(self.lower_limit)
            self._erf_lower = sp.erf((log_lower_limit - self.log_mean) / (np.sqrt(2) * self.sigma_ln))
        if np.isinf(self.upper_limit):
            self._erf_upper = 1.0
        else:
            log_upper_limit = np.log(self.upper_limit)
            self._erf_upper = sp.erf((log_upper_limit - self.log_mean) / (np.sqrt(2) * self.sigma_ln))

    def value_for(self, unit):
        '''
        Map a uniformly distributed value between 0 and 1 to the distribution
        represented by this prior.
        '''
        # Map `unit` from [0,1] to the range described by the upper and lower limits
        erf_mapped = self._erf_lower + unit * (self._erf_upper - self._erf_lower)
        log_value = self.log_mean + sp.erfinv(erf_mapped) * np.sqrt(2) * self.sigma_ln
        return np.exp(log_value)

    def logpdf(self, value):
        '''
        The log P(value)
        '''
        log_val = np.log(value)
        # The normalization of a gaussian is 1/( sqrt(2 pi) sigma)
        gaussian_norm = - np.log(2 * np.pi)/2 - np.log(self.sigma_ln)
        norm = gaussian_norm - np.log((self._erf_upper - self._erf_lower) / 2)
        # The log pdf of this log normal distribution is just that of the normal dist
        # minus log val due to the coordinate transform
        raw_log_pdf = - (log_val - self.log_mean)**2 / (2 * self.sigma_ln**2) - log_val

        log_pdf = np.array(norm + raw_log_pdf)
        log_pdf[(np.array(value) <= self.lower_limit) | (np.array(value) >= self.upper_limit)] = -np.inf
        return log_pdf
