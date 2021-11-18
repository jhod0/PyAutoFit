from typing import Optional

from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.samples import OptimizerSamples, Sample

import numpy as np


class LBFGSSamples(OptimizerSamples):

    def __init__(
            self,
            model: AbstractPriorModel,
            x0: np.ndarray,
            log_posterior_list: np.ndarray,
            total_iterations: int,
            time: Optional[float] = None,
    ):
        """
        Create an *OptimizerSamples* object from this non-linear search's output files on the hard-disk and model.

        For LBFGS, all quantities are extracted via pickled states of the particle and cost histories.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space. This maps the points from unit
            cube values to physical values via the priors.
        """

        self.x0 = x0
        self._log_posterior_list = log_posterior_list
        self.total_iterations = total_iterations

        parameter_lists = [list(self.x0)]
        log_prior_list = model.log_prior_list_from(parameter_lists=parameter_lists)
        log_likelihood_list = [lp - prior for lp, prior in zip(self._log_posterior_list, log_prior_list)]
        weight_list = len(log_likelihood_list) * [1.0]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list
        )

        super().__init__(
            model=model,
            sample_list=sample_list,
            time=time,
        )