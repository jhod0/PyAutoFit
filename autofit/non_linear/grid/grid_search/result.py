from typing import List, Union, Iterable

import numpy as np

from autofit import exc
from autofit.mapper import model_mapper as mm
from autofit.mapper.prior.abstract import Prior

from autofit.non_linear.samples.interface import SamplesInterface

LimitLists = List[List[float]]


class GridSearchResult:
    def __init__(
        self,
        samples: List[SamplesInterface],
        lower_limits_lists: LimitLists,
        grid_priors: List[Prior],
    ):
        """
        The sample of a grid search.

        Parameters
        ----------
        samples
            The samples of the non linear optimizations performed at each grid step
        lower_limits_lists
            A list of lists of values representing the lower bounds of the grid searched values at each step
        """
        self.lower_limits_lists = lower_limits_lists
        self.samples = samples
        self.no_dimensions = len(self.lower_limits_lists[0])
        self.no_steps = len(self.lower_limits_lists)
        self.side_length = int(self.no_steps ** (1 / self.no_dimensions))
        self.step_size = 1 / self.side_length
        self.grid_priors = grid_priors

    @property
    def physical_lower_limits_lists(self) -> LimitLists:
        """
        The lower physical values for each grid square
        """
        return self._physical_values_for(self.lower_limits_lists)

    @property
    def physical_centres_lists(self) -> LimitLists:
        """
        The middle physical values for each grid square
        """
        return self._physical_values_for(self.centres_lists)

    @property
    def physical_upper_limits_lists(self) -> LimitLists:
        """
        The upper physical values for each grid square
        """
        return self._physical_values_for(self.upper_limits_lists)

    @property
    def upper_limits_lists(self) -> LimitLists:
        """
        The upper values for each grid square
        """
        return [
            [limit + self.step_size for limit in limits]
            for limits in self.lower_limits_lists
        ]

    @property
    def centres_lists(self) -> LimitLists:
        """
        The centre values for each grid square
        """
        return [
            [(upper + lower) / 2 for upper, lower in zip(upper_limits, lower_limits)]
            for upper_limits, lower_limits in zip(
                self.lower_limits_lists, self.upper_limits_lists
            )
        ]

    def _physical_values_for(self, unit_lists: LimitLists) -> LimitLists:
        """
        Compute physical values for lists of lists of unit hypercube
        values.

        Parameters
        ----------
        unit_lists
            A list of lists of hypercube values

        Returns
        -------
        A list of lists of physical values
        """
        return [
            [prior.value_for(limit) for prior, limit in zip(self.grid_priors, limits)]
            for limits in unit_lists
        ]

    def __setstate__(self, state):
        return self.__dict__.update(state)

    def __getstate__(self):
        return self.__dict__

    def __getattr__(self, item: str) -> object:
        """
        We default to getting attributes from the best sample. This allows promises to reference best samples.
        """
        return getattr(self.best_samples, item)

    @property
    def shape(self):
        return self.no_dimensions * (int(self.no_steps ** (1 / self.no_dimensions)),)

    @property
    def best_samples(self):
        """
        The best sample of the grid search. That is, the sample output by the non linear search that had the highest
        maximum figure of merit.

        Returns
        -------
        best_sample: sample
        """
        return max(
            self.samples,
            key=lambda sample: sample.log_likelihood,
        )

    @property
    def best_model(self):
        """
        Returns
        -------
        best_model: mm.ModelMapper
            The model mapper instance associated with the highest figure of merit from the grid search
        """
        return self.best_sample.model

    @property
    def all_models(self):
        """
        Returns
        -------
        all_models: [mm.ModelMapper]
            All model mapper instances used in the grid search
        """
        return [sample.model for sample in self.samples]

    @property
    def physical_step_sizes(self):
        physical_step_sizes = []

        # TODO : Make this work for all dimensions in a less ugly way.

        for dim in range(self.no_dimensions):
            values = [value[dim] for value in self.physical_lower_limits_lists]
            diff = [abs(values[n] - values[n - 1]) for n in range(1, len(values))]

            if dim == 0:
                physical_step_sizes.append(np.max(diff))
            elif dim == 1:
                physical_step_sizes.append(np.min(diff))
            else:
                raise exc.GridSearchException(
                    "This feature does not support > 2 dimensions"
                )

        return tuple(physical_step_sizes)

    def _list_to_native(self, lst: List):
        return np.reshape(np.array(lst), self.shape)

    @property
    def samples_native(self):
        """
        The sample of every grid search on a NumPy array whose shape is the native dimensions of the grid search.

        For example, for a 2x2 grid search the shape of the Numpy array is (2,2) and it is numerically ordered such
        that the first search's sample (corresponding to unit priors (0.0, 0.0)) are in the first value (E.g. entry
        [0, 0]) of the NumPy array.
        """
        return self._list_to_native(lst=[sample for sample in self.samples])

    def attribute_grid(self, attribute_path: Union[str, Iterable[str]]) -> np.ndarray:
        """
        Get a list of the attribute of the best instance from every search in a numpy array with the native dimensions
        of the grid search.

        Parameters
        ----------
        attribute_path
            The path to the attribute to get from the instance

        Returns
        -------
        A numpy array of the attribute of the best instance from every search in the grid search.
        """
        if isinstance(attribute_path, str):
            attribute_path = attribute_path.split(".")

        def _get_attribute(
            obj: Union[np.ndarray, SamplesInterface],
        ):
            if isinstance(obj, np.ndarray):
                return np.array([_get_attribute(item) for item in obj])

            item = obj.instance
            for attribute_name in attribute_path:
                item = getattr(item, attribute_name)
            return item

        return _get_attribute(self.samples_native)

    @property
    def log_likelihoods_native(self):
        """
        The maximum log likelihood of every grid search on a NumPy array whose shape is the native dimensions of the
        grid search.

        For example, for a 2x2 grid search the shape of the Numpy array is (2,2) and it is numerically ordered such
        that the first search's maximum likelihood (corresponding to unit priors (0.0, 0.0)) are in the first
        value (E.g. entry [0, 0]) of the NumPy array.
        """
        return self._list_to_native(
            lst=[sample.log_likelihood for sample in self.samples]
        )

    @property
    def log_evidences_native(self):
        """
        The log evidence of every grid search on a NumPy array whose shape is the native dimensions of the grid search.

        For example, for a 2x2 grid search the shape of the Numpy array is (2,2) and it is numerically ordered such
        that the first search's log evidence (corresponding to unit priors (0.0, 0.0)) are in the first value (E.g.
        entry [0, 0]) of the NumPy array.
        """
        return self._list_to_native(
            lst=[samples.log_evidence for samples in self.samples]
        )
