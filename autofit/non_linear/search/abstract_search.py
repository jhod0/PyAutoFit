import copy
import logging
import multiprocessing as mp
import os
import time
import warnings
from abc import ABC, abstractmethod
from collections import Counter
from functools import wraps
from os import path
from pathlib import Path
from typing import Optional, Union, Tuple, List, Dict

import numpy as np

from autoconf import conf, cached_property
from autofit import exc
from autofit.database.sqlalchemy_ import sa
from autofit.graphical import (
    MeanField,
    AnalysisFactor,
    _HierarchicalFactor,
    FactorApproximation,
)
from autofit.graphical.utils import Status
from autofit.mapper.prior_model.collection import Collection
from autofit.non_linear.initializer import Initializer
from autofit.non_linear.parallel import SneakyPool
from autofit.non_linear.paths.abstract import AbstractPaths
from autofit.non_linear.paths.directory import DirectoryPaths
from autofit.non_linear.paths.sub_directory_paths import SubDirectoryPaths
from autofit.non_linear.samples.samples import Samples
from autofit.non_linear.result import Result
from autofit.non_linear.timer import Timer
from autofit.non_linear.analysis import Analysis
from autofit.non_linear.analysis.combined import CombinedResult
from autofit.non_linear.analysis.indexed import IndexCollectionAnalysis
from autofit.non_linear.paths.null import NullPaths
from autofit.graphical.declarative.abstract import PriorFactor
from autofit.graphical.expectation_propagation import AbstractFactorOptimiser

logger = logging.getLogger(__name__)


def check_cores(func):
    """
    Checks how many cores the search has been configured to
    use and then returns None instead of calling the pool
    creation function in the case that only one core has
    been set.

    Parameters
    ----------
    func
        A function that creates a pool

    Returns
    -------
    None or a pool
    """

    @wraps(func)
    def wrapper(self, *args, **kwargs):
        self.logger.info(f"number_of_cores == {self.number_of_cores}...")
        if self.number_of_cores == 1:
            self.logger.info("...not using pool")
            return None
        return func(self, *args, **kwargs)

    return wrapper


class NonLinearSearch(AbstractFactorOptimiser, ABC):
    def __init__(
        self,
        name: Optional[str] = None,
        path_prefix: Optional[str] = None,
        unique_tag: Optional[str] = None,
        initializer: Initializer = None,
        iterations_per_update: int = None,
        number_of_cores: int = 1,
        session: Optional[sa.orm.Session] = None,
        **kwargs,
    ):
        """
        Abstract base class for non-linear searches.L

        This class sets up the file structure for the non-linear search, which are standardized across all non-linear
        searches.

        Parameters
        ----------
        name
            The name of the search, controlling the last folder results are output.
        path_prefix
            The path of folders prefixing the name folder where results are output.
        unique_tag
            The name of a unique tag for this model-fit, which will be given a unique entry in the sqlite database
            and also acts as the folder after the path prefix and before the search name.
        initializer
            Generates the initialize samples of non-linear parameter space (see autofit.non_linear.initializer).
        session
            An SQLAlchemy session instance so the results of the model-fit are written to an SQLite database.
        """
        super().__init__()

        from autofit.non_linear.paths.database import DatabasePaths

        if name:
            path_prefix = Path(path_prefix or "")
        self.path_prefix = path_prefix

        self.path_prefix_no_unique_tag = path_prefix

        self._logger = None

        logger.info(f"Creating search")

        if unique_tag is not None and path_prefix is not None:
            path_prefix = path_prefix / unique_tag

        self.unique_tag = unique_tag

        if session is not None:
            logger.debug("Session found. Using database.")
            paths = DatabasePaths(
                name=name,
                path_prefix=path_prefix,
                session=session,
                save_all_samples=kwargs.get("save_all_samples", False),
                unique_tag=unique_tag,
            )
        elif name is not None or path_prefix:
            logger.debug("Session not found. Using directory output.")
            paths = DirectoryPaths(
                name=name, path_prefix=path_prefix, unique_tag=unique_tag
            )
        else:
            paths = NullPaths()

        self.paths: AbstractPaths = paths

        self.force_pickle_overwrite = conf.instance["general"]["output"][
            "force_pickle_overwrite"
        ]
        self.skip_save_samples = kwargs.get("skip_save_samples")
        if self.skip_save_samples is None:
            self.skip_save_samples = conf.instance["general"]["output"].get(
                "skip_save_samples"
            )

        self.force_visualize_overwrite = conf.instance["general"]["output"][
            "force_visualize_overwrite"
        ]

        if initializer is None:
            self.logger.debug("Creating initializer ")
            self.initializer = Initializer.from_config(config=self._config)
        else:
            self.initializer = initializer

        self.iterations_per_update = iterations_per_update or self._config(
            "updates", "iterations_per_update"
        )

        if conf.instance["general"]["hpc"]["hpc_mode"]:
            self.iterations_per_update = conf.instance["general"]["hpc"][
                "iterations_per_update"
            ]

        self.remove_state_files_at_end = self._config(
            "updates",
            "remove_state_files_at_end",
        )

        self.iterations = 0

        self.should_profile = conf.instance["general"]["profiling"]["should_profile"]

        self.silence = self._config("printing", "silence")

        if conf.instance["general"]["hpc"]["hpc_mode"]:
            self.silence = True

        self.kwargs = kwargs

        for key, value in self.config_dict_search.items():
            setattr(self, key, value)

        try:
            for key, value in self.config_dict_run.items():
                setattr(self, key, value)
        except KeyError:
            pass

        self.number_of_cores = number_of_cores

        if number_of_cores > 1 and any(
            os.environ.get(key) != "1"
            for key in (
                "OPENBLAS_NUM_THREADS",
                "MKL_NUM_THREADS",
                "OMP_NUM_THREADS",
                "VECLIB_MAXIMUM_THREADS",
                "NUMEXPR_NUM_THREADS",
            )
        ):
            warnings.warn(
                exc.SearchWarning(
                    """
                    The non-linear search is using multiprocessing (number_of_cores>1). 

                    However, the following environment variables have not been set to 1:

                    OPENBLAS_NUM_THREADS
                    MKL_NUM_THREADS
                    OMP_NUM_THREADS
                    VECLIB_MAXIMUM_THREADS
                    NUMEXPR_NUM_THREADS

                    This can lead to performance issues, because both the non-linear search and libraries that may be
                    used in your `log_likelihood_function` evaluation (e.g. NumPy, SciPy, scikit-learn) may attempt to
                    parallelize over all cores available.

                    This will lead to slow-down, due to overallocation of tasks over the CPUs.

                    To mitigate this, set the environment variables to 1 via the following command on your
                    bash terminal / command line:

                    export OPENBLAS_NUM_THREADS=1
                    export MKL_NUM_THREADS=1
                    export OMP_NUM_THREADS=1
                    export VECLIB_MAXIMUM_THREADS=1
                    export NUMEXPR_NUM_THREADS=1

                    This means only the non-linear search is parallelized over multiple cores.

                    If you "know what you are doing" and do not want these environment variables to be set to one, you 
                    can disable this warning by changing the following entry in the config files:

                    `config -> general.yaml -> parallel: -> warn_environment_variable=False`
                    """
                )
            )

        self.optimisation_counter = Counter()

    __identifier_fields__ = tuple()

    def optimise(
        self,
        factor_approx: FactorApproximation,
        status: Status = Status(),
    ) -> Tuple[MeanField, Status]:
        """
        Perform optimisation for expectation propagation. Currently only
        applicable for ModelFactors created by the declarative interface.

        1. Analysis and model classes are extracted from the factor.
        2. Priors are updated from the mean field.
        3. Analysis and model are fit as usual.
        4. A new mean field is constructed with the (posterior) 'linking' priors.
        5. Projection is performed to produce an updated EPMeanField object.

        Output directories are generated according to the factor and the number
        of the search. For example a factor called "factor" would output:

        factor/optimization_0/<identifier>
        factor/optimization_1/<identifier>
        factor/optimization_2/<identifier>

        For the first, second and third optimizations respectively.

        Parameters
        ----------
        factor_approx
            A collection of messages defining the current best approximation to
            some global model
        status

        Returns
        -------
        An updated approximation to the model having performed optimisation on
        a single factor.
        """

        factor = factor_approx.factor

        _ = status
        if not isinstance(factor, (AnalysisFactor, PriorFactor, _HierarchicalFactor)):
            raise NotImplementedError(
                f"Optimizer {self.__class__.__name__} can only be applied to"
                f" AnalysisFactors, HierarchicalFactors and PriorFactors"
            )

        model = factor.prior_model.mapper_from_prior_arguments(
            {
                prior: prior.with_message(message)
                for prior, message in factor_approx.cavity_dist.arguments.items()
            }
        )

        analysis = factor.analysis

        number = self.optimisation_counter[factor.name]

        self.optimisation_counter[factor.name] += 1

        self.paths = SubDirectoryPaths(
            parent=self.paths,
            analysis_name=f"{factor.name}/optimization_{number}",
            is_flat=True,
        )

        result = self.fit(model=model, analysis=analysis)

        new_model_dist = MeanField.from_priors(result.projected_model.priors)

        status.result = result

        return new_model_dist, status

    @property
    def name(self):
        return self.paths.name

    def __getstate__(self):
        """
        Remove the logger for pickling
        """
        state = self.__dict__.copy()
        if "_logger" in state:
            del state["_logger"]
        if "paths" in state:
            del state["paths"]
        return state

    @property
    def logger(self):
        if not hasattr(self, "_logger"):
            self._logger = None
        if self._logger is None:
            logger_ = logging.getLogger(self.name)
            self._logger = logger_
        return self._logger

    @property
    def timer(self):
        return Timer(self.paths.search_internal_path)

    @property
    def paths(self) -> Optional[AbstractPaths]:
        return self._paths

    @paths.setter
    def paths(self, paths: Optional[AbstractPaths]):
        if paths is not None:
            paths.search = self
        self._paths = paths

    def copy_with_paths(self, paths):
        self.logger.debug(f"Creating a copy of {self._paths.name}")
        search_instance = copy.copy(self)
        search_instance.paths = paths
        search_instance._logger = None

        return search_instance

    class Fitness:
        def __init__(
            self, paths, model, analysis, samples_from_model, log_likelihood_cap=None
        ):
            self.i = 0

            self.paths = paths
            self.analysis = analysis

            self.model = model
            self.samples_from_model = samples_from_model

            self.log_likelihood_cap = log_likelihood_cap

        def __call__(self, parameters, *kwargs):
            try:
                figure_of_merit = self.figure_of_merit_from(parameter_list=parameters)

                if np.isnan(figure_of_merit):
                    return self.resample_figure_of_merit

                return figure_of_merit

            except exc.FitException:
                return self.resample_figure_of_merit

        def fit_instance(self, instance):
            log_likelihood = self.analysis.log_likelihood_function(instance=instance)

            if self.log_likelihood_cap is not None:
                if log_likelihood > self.log_likelihood_cap:
                    log_likelihood = self.log_likelihood_cap

            return log_likelihood

        def log_likelihood_from(self, parameter_list):
            instance = self.model.instance_from_vector(vector=parameter_list)
            log_likelihood = self.fit_instance(instance)

            return log_likelihood

        def log_posterior_from(self, parameter_list):
            log_likelihood = self.log_likelihood_from(parameter_list=parameter_list)
            log_prior_list = self.model.log_prior_list_from_vector(
                vector=parameter_list
            )

            return log_likelihood + sum(log_prior_list)

        def figure_of_merit_from(self, parameter_list):
            """
            The figure of merit is the value that the `NonLinearSearch` uses to sample parameter space. This varies
            between different `NonLinearSearch`s, for example:

                - The *Optimizer* *PySwarms* uses the chi-squared value, which is the -2.0*log_posterior.
                - The *MCMC* algorithm *Emcee* uses the log posterior.
                - Nested samplers such as *Dynesty* use the log likelihood.
            """
            raise NotImplementedError()

        @staticmethod
        def prior(cube, model):
            # NEVER EVER REFACTOR THIS LINE! Haha.

            phys_cube = model.vector_from_unit_vector(unit_vector=cube)

            for i in range(len(phys_cube)):
                cube[i] = phys_cube[i]

            return cube

        @staticmethod
        def fitness(cube, model, fitness):
            return fitness(instance=model.instance_from_vector(cube))

        @property
        def samples(self):
            return self.samples_from_model(model=self.model)

        @property
        def resample_figure_of_merit(self):
            """
            If a sample raises a FitException, this value is returned to signify that the point requires resampling or
             should be given a likelihood so low that it is discard.
            """
            return -np.inf

    def fit_sequential(
        self,
        model,
        analysis: IndexCollectionAnalysis,
        info=None,
        pickle_files=None,
        log_likelihood_cap=None,
    ) -> CombinedResult:
        """
        Fit multiple analyses contained within the analysis sequentially.

        This can be useful for avoiding very high dimensional parameter spaces.

        Parameters
        ----------
        log_likelihood_cap
        analysis
            Multiple analyses that are fit sequentially
        model
            An object that represents possible instances of some model with a
            given dimensionality which is the number of free dimensions of the
            model.
        info
            Optional dictionary containing information about the fit that can be loaded by the aggregator.
        pickle_files : [str]
            Optional list of strings specifying the path and filename of .pickle files, that are copied to each
            model-fits pickles folder so they are accessible via the Aggregator.

        Returns
        -------
        An object combining the results of each individual optimisation.

        Raises
        ------
        AssertionError
            If the model has 0 dimensions.
        ValueError
            If the analysis is not a combined analysis
        """
        results = []

        _paths = self.paths
        original_name = self.paths.name or "analysis"

        model = analysis.modify_model(model=model)

        try:
            if not isinstance(model, Collection):
                model = [model for _ in range(len(analysis.analyses))]
        except AttributeError:
            raise ValueError(
                f"Analysis with type {type(analysis)} is not supported by fit_sequential"
            )

        for i, (model, analysis) in enumerate(zip(model, analysis.analyses)):
            self.paths = copy.copy(_paths)
            self.paths.name = f"{original_name}/{i}"
            results.append(
                self.fit(
                    model=model,
                    analysis=analysis,
                    info=info,
                    pickle_files=pickle_files,
                    log_likelihood_cap=log_likelihood_cap,
                )
            )
        self.paths = _paths
        return CombinedResult(results)

    def fit(
        self,
        model,
        analysis: "Analysis",
        info=None,
        pickle_files=None,
        log_likelihood_cap=None,
        bypass_nuclear_if_on: bool = False,
    ) -> Union["Result", List["Result"]]:
        """
        Fit a model, M with some function f that takes instances of the
        class represented by model M and gives a score for their fitness.

        A model which represents possible instances with some dimensionality is fit.

        The analysis provides two functions. One visualises an instance of a model and the
        other scores an instance based on how well it fits some data. The search
        produces instances of the model by picking points in an N dimensional space.

        Parameters
        ----------
        log_likelihood_cap
        analysis : af.Analysis
            An object that encapsulates the data and a log likelihood function.
        model : ModelMapper
            An object that represents possible instances of some model with a
            given dimensionality which is the number of free dimensions of the
            model.
        info : dict
            Optional dictionary containing information about the fit that can be loaded by the aggregator.
        pickle_files : [str]
            Optional list of strings specifying the path and filename of .pickle files, that are copied to each
            model-fits pickles folder so they are accessible via the Aggregator.
        bypass_nuclear_if_on
            If nuclear mode is on (environment variable "PYAUTOFIT_NUCLEAR_MODE=1") passing this as True will
            bypass it.

        Returns
        -------
        An object encapsulating how well the model fit the data, the best fit instance
        and an updated model with free parameters updated to represent beliefs
        produced by this fit.

        Raises
        ------
        AssertionError
            If the model has 0 dimensions.
        """
        self.check_model(model=model)

        self.logger.info("Starting search")

        model = analysis.modify_model(model)
        self.paths.model = model
        self.paths.unique_tag = self.unique_tag
        self.paths.restore()

        analysis = analysis.modify_before_fit(paths=self.paths, model=model)

        if analysis.should_visualize(paths=self.paths):
            analysis.visualize_before_fit(
                paths=self.paths,
                model=model,
            )
            analysis.visualize_before_fit_combined(
                analyses=None,
                paths=self.paths,
                model=model,
            )

        if not self.paths.is_complete or self.force_pickle_overwrite:
            self.logger.info("Saving path info")

            self.paths.save_all(
                search_config_dict=self.config_dict_search,
                info=info,
                pickle_files=pickle_files,
            )
            analysis.save_attributes(paths=self.paths)

        if not self.paths.is_complete:
            self.logger.info("Not complete. Starting non-linear search.")

            self.timer.start()

            model.freeze()
            self._fit(
                model=model, analysis=analysis, log_likelihood_cap=log_likelihood_cap
            )
            model.unfreeze()

            self.paths.completed()

            samples = self.perform_update(
                model=model, analysis=analysis, during_analysis=False
            )

            result = analysis.make_result(
                samples=samples,
            )

            analysis.save_results(paths=self.paths, result=result)

            if not self.skip_save_samples:
                self.paths.save_json("samples_summary", samples.summary().dict())

        else:
            self.logger.info(f"Already completed, skipping non-linear search.")

            try:
                samples = self.samples_from(model=model)
            except (FileNotFoundError, NotImplementedError):
                samples = self.paths.load_object(name="samples")

            if self.force_visualize_overwrite:
                self.perform_visualization(
                    model=model, analysis=analysis, during_analysis=False
                )

            result = analysis.make_result(
                samples=samples,
            )

            if self.force_pickle_overwrite:
                self.logger.info("Forcing pickle overwrite")
                if not self.skip_save_samples:
                    self.paths.save_json("samples_summary", samples.summary().dict())

                try:
                    self.paths.save_object("results", samples.results)
                except AttributeError:
                    self.paths.save_object("results", samples.results_internal)

                analysis.save_results(paths=self.paths, result=result)

        analysis = analysis.modify_after_fit(
            paths=self.paths, model=model, result=result
        )

        self.logger.info("Removing zip file")
        self.paths.zip_remove()

        if not bypass_nuclear_if_on:
            self.paths.zip_remove_nuclear()

        return result

    @abstractmethod
    def _fit(self, model, analysis, log_likelihood_cap=None):
        pass

    def check_model(self, model):
        if model is not None and model.prior_count == 0:
            raise AssertionError("Model has no priors! Cannot fit a 0 dimension model.")

    def config_dict_with_test_mode_settings_from(self, config_dict: Dict) -> Dict:
        return config_dict

    @property
    def _class_config(self) -> Dict:
        return self.config_type[self.__class__.__name__]

    @cached_property
    def config_dict_search(self) -> Dict:
        config_dict = copy.deepcopy(self._class_config["search"])

        for key, value in config_dict.items():
            try:
                config_dict[key] = self.kwargs[key]
            except KeyError:
                pass

        return config_dict

    @cached_property
    def config_dict_run(self) -> Dict:
        config_dict = copy.deepcopy(self._class_config["run"])

        for key, value in config_dict.items():
            try:
                config_dict[key] = self.kwargs[key]
            except KeyError:
                pass

        if os.environ.get("PYAUTOFIT_TEST_MODE") == "1":
            logger.warning(f"TEST MODE ON: SEARCH WILL SKIP SAMPLING\n\n")

            config_dict = self.config_dict_with_test_mode_settings_from(
                config_dict=config_dict
            )

        return config_dict

    @property
    def config_dict_settings(self) -> Dict:
        return self._class_config["settings"]

    @property
    def config_type(self):
        raise NotImplementedError()

    def _config(self, section, attribute_name):
        """
        Get a config field from this search's section in non_linear.ini by a key and value type.

        Parameters
        ----------
        attribute_name: str
            The analysis_path of the field

        Returns
        -------
        attribute
            An attribute for the key with the specified type.
        """
        return self._class_config[section][attribute_name]

    def perform_update(
        self, model: Collection, analysis: Analysis, during_analysis: bool
    ):
        """
        Perform an update of the non-linear search's model-fitting results.

        This occurs every `iterations_per_update` of the non-linear search and once it is complete.

        The update performs the following tasks (if the settings indicate they should be performed):

        1) Visualize the search results (e.g. a cornerplot).
        2) Visualize the maximum log likelihood model using model-specific visualization implented via the `Analysis`
           object.
        3) Perform profiling of the analysis object `log_likelihood_function` and ouptut run-time information.
        4) Output the `search.summary` file which contains information on model-fitting so far.
        5) Output the `model.results` file which contains a concise text summary of the model results so far.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space.
        analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the `NonLinearSearch` maximizes.
        during_analysis
            If the update is during a non-linear search, in which case tasks are only performed after a certain number
            of updates and only a subset of visualization may be performed.
        """

        self.iterations += self.iterations_per_update
        self.logger.info(
            f"{self.iterations} Iterations: Performing update (Visualization, outputting samples, etc.)."
        )

        self.timer.update()

        samples = self.samples_from(model=model)

        self.paths.samples_to_csv(samples=samples)

        try:
            instance = samples.max_log_likelihood()
        except exc.FitException:
            return samples

        self.perform_visualization(
            model=model, analysis=analysis, during_analysis=during_analysis
        )

        if self.should_profile:
            self.logger.debug("Profiling Maximum Likelihood Model")
            analysis.profile_log_likelihood_function(
                paths=self.paths,
                instance=instance,
            )

        self.logger.debug("Outputting model result")
        try:
            start = time.time()
            analysis.log_likelihood_function(instance=instance)
            log_likelihood_function_time = time.time() - start

            self.paths.save_summary(
                samples=samples,
                log_likelihood_function_time=log_likelihood_function_time,
            )
        except exc.FitException:
            pass

        if not during_analysis and self.remove_state_files_at_end:
            self.logger.debug("Removing state files")
            try:
                self.remove_state_files()
            except FileNotFoundError:
                pass

        return samples

    def perform_visualization(self, model, analysis, during_analysis):
        """
        Perform visualization of the non-linear search's model-fitting results.

        This occurs every `iterations_per_update` of the non-linear search, when the search is complete and can
        also be forced to occur even though a search is completed on a rerun, to update the visualization
        with different `matplotlib` settings.

        The update performs the following tasks (if the settings indicate they should be performed):

        1) Visualize the search results (e.g. a cornerplot).
        2) Visualize the maximum log likelihood model using model-specific visualization implented via the `Analysis`
           object.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space.
        analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data, returning
            the log likelihood the `NonLinearSearch` maximizes.
        during_analysis
            If the update is during a non-linear search, in which case tasks are only performed after a certain number
            of updates and only a subset of visualization may be performed.
        """
        try:
            samples = self.samples_from(model=model)
        except FileNotFoundError:
            samples = self.paths.load_object(name="samples")

        try:
            instance = samples.max_log_likelihood()
        except exc.FitException:
            return samples

        if analysis.should_visualize(paths=self.paths, during_analysis=during_analysis):
            if not isinstance(self.paths, NullPaths):
                self.plot_results(samples=samples)

        self.logger.debug("Visualizing")
        if analysis.should_visualize(paths=self.paths, during_analysis=during_analysis):
            analysis.visualize(
                paths=self.paths, instance=instance, during_analysis=during_analysis
            )
            analysis.visualize_combined(
                analyses=None,
                paths=self.paths,
                instance=instance,
                during_analysis=during_analysis,
            )

    @property
    def samples_cls(self):
        raise NotImplementedError()

    def remove_state_files(self):
        pass

    def samples_from(self, model) -> Samples:
        """
        Loads the samples of a non-linear search from its output files.

        The samples can be loaded from one of two files, which are attempted to be loading in the following order:

        1) Load via the internal results of the non-linear search, which are specified to that search's outputs
           (e.g. the .hdf file output by the MCMC method `emcee`).

        2) Load via the `samples.csv` and `samples_info.json` files of the search, which are outputs that are the
           same for all non-linear searches as they are homogenized by autofit.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space.
        """
        try:
            return self.samples_via_internal_from(model=model)
        except (FileNotFoundError, NotImplementedError):
            return self.samples_via_csv_from(model=model)

    def samples_via_internal_from(self, model):
        raise NotImplementedError

    def samples_via_csv_from(self, model) -> Samples:
        """
        Returns a `Samples` object from the `samples.csv` and `samples_info.json` files.

        The samples contain all information on the parameter space sampling (e.g. the parameters,
        log likelihoods, etc.).

        The samples in csv format are already converted to the autofit format, where samples are lists of values
        (e.g. `parameter_lists`, `log_likelihood_list`).

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        """

        return self.samples_cls.from_csv(
            paths=self.paths,
            model=model,
        )

    @check_cores
    def make_pool(self):
        """Make the pool instance used to parallelize a `NonLinearSearch` alongside a set of unique ids for every
        process in the pool. If the specified number of cores is 1, a pool instance is not made and None is returned.

        The pool cannot be set as an attribute of the class itself because this prevents pickling, thus it is generated
        via this function before calling the non-linear search.

        The pool instance is also set up with a list of unique pool ids, which are used during model-fitting to
        identify a 'master core' (the one whose id value is lowest) which handles model result output, visualization,
        etc."""
        self.logger.info("...using pool")
        return mp.Pool(processes=self.number_of_cores)

    @check_cores
    def make_sneaky_pool(self, fitness: Fitness) -> Optional[SneakyPool]:
        """
        Create a pool for multiprocessing that uses slight-of-hand
        to avoid copying the fitness function between processes
        multiple times.

        Parameters
        ----------
        fitness
            An instance of a fitness class used to evaluate the
            likelihood that a particular model is correct

        Returns
        -------
        An implementation of a multiprocessing pool
        """
        self.logger.warning(
            "...using SneakyPool. This copies the likelihood function "
            "to each process on instantiation to avoid copying multiple "
            "times."
        )
        return SneakyPool(
            processes=self.number_of_cores, paths=self.paths, fitness=fitness
        )

    def __eq__(self, other):
        return isinstance(other, NonLinearSearch) and self.__dict__ == other.__dict__

    def plot_results(self, samples):
        pass