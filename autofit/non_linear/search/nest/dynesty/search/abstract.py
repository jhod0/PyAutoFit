import os
from abc import ABC
from os import path
from pathlib import Path
from typing import Optional, Tuple, Union

import numpy as np
from dynesty import NestedSampler, DynamicNestedSampler

from autoconf import conf
from autofit import exc
from autofit.database.sqlalchemy_ import sa
from autofit.mapper.prior_model.abstract import AbstractPriorModel
from autofit.non_linear.search.nest.abstract_nest import AbstractNest
from autofit.non_linear.samples.sample import Sample
from autofit.non_linear.samples.nest import SamplesNest
from autofit.plot.output import Output

def prior_transform(cube, model):
    phys_cube = model.vector_from_unit_vector(
        unit_vector=cube, ignore_prior_limits=True
    )

    for i in range(len(phys_cube)):
        cube[i] = phys_cube[i]

    return cube


class AbstractDynesty(AbstractNest, ABC):
    def __init__(
            self,
            name: str = "",
            path_prefix: str = "",
            unique_tag: Optional[str] = None,
            iterations_per_update: int = None,
            number_of_cores: int = None,
            session: Optional[sa.orm.Session] = None,
            **kwargs,
    ):
        """
        A Dynesty non-linear search.

        For a full description of Dynesty, checkout its GitHub and readthedocs webpages:

        https://github.com/joshspeagle/dynesty
        https://dynesty.readthedocs.io/en/latest/index.html

        Parameters
        ----------
        name
            The name of the search, controlling the last folder results are output.
        path_prefix
            The path of folders prefixing the name folder where results are output.
        unique_tag
            The name of a unique tag for this model-fit, which will be given a unique entry in the sqlite database
            and also acts as the folder after the path prefix and before the search name.
        iterations_per_update
            The number of iterations performed between every Dynesty back-up (via dumping the Dynesty instance as a
            pickle).
        number_of_cores
            The number of cores Emcee sampling is performed using a Python multiprocessing Pool instance. If 1, a
            pool instance is not created and the job runs in serial.
        session
            An SQLalchemy session instance so the results of the model-fit are written to an SQLite database.
        """

        number_of_cores = (
            self._config("parallel", "number_of_cores")
            if number_of_cores is None
            else number_of_cores
        )

        super().__init__(
            name=name,
            path_prefix=path_prefix,
            unique_tag=unique_tag,
            iterations_per_update=iterations_per_update,
            number_of_cores=number_of_cores,
            session=session,
            **kwargs,
        )

        self.logger.debug(f"Creating {self.__class__.__name__} Search")

    class Fitness(AbstractNest.Fitness):
        @property
        def resample_figure_of_merit(self):
            """
            If a sample raises a FitException, this value is returned to signify that the point requires resampling or
            should be given a likelihood so low that it is discard.

            -np.inf is an invalid sample value for Dynesty, so we instead use a large negative number.
            """
            return -1.0e99

        def history_save(self):
            pass

    def _fit(
            self,
            model: AbstractPriorModel,
            analysis,
            log_likelihood_cap: Optional[float] = None,
    ):
        """
        Fit a model using Dynesty and the Analysis class which contains the data and returns the log likelihood from
        instances of the model, which the `NonLinearSearch` seeks to maximize.

        By default, Dynesty runs using an in-built multiprocessing Pool option. This occurs even
        if `number_of_cores=1`, because the dynesty savestate includes this pool, meaning that a resumed run can
        then increase the `number_of_cores`.

        However, certain operating systems (e.g. Windows) do not support Python multiprocessing particularly well.
        This can cause Dynesty to crash when a pool is included. If this occurs (raising a `RunTimeException`)
        a Dynesty object without a pool is created and used instead.

        Parameters
        ----------
        model
            The model which generates instances for different points in parameter space.
        analysis
            Contains the data and the log likelihood function which fits an instance of the model to the data,
            returning the log likelihood dynesty maximizes.
        log_likelihood_cap
            An optional cap to the log likelihood values, which means all likelihood evaluations above this value
            are rounded down to it. This is used to remove numerical instability in an Astronomy based project.

        Returns
        -------
        A result object comprising the Samples object that includes the maximum log likelihood instance and full
        set of accepted samples of the fit.
        """

        from dynesty.pool import Pool

        fitness = self.Fitness(
            paths=self.paths,
            model=model,
            analysis=analysis,
            samples_from_model=self.samples_from,
            log_likelihood_cap=log_likelihood_cap,
        )

        if os.path.exists(self.checkpoint_file):
            self.logger.info(
                "Existing Dynesty samples found, resuming non-linear search."
            )
        else:
            self.logger.info(
                "No Dynesty samples found, beginning new non-linear search. "
            )

        finished = False

        while not finished:

            checkpoint_exists = os.path.exists(self.checkpoint_file)

            try:

                if conf.instance["non_linear"]["nest"][self.__class__.__name__][
                    "parallel"
                ].get("force_x1_cpu") or self.kwargs.get("force_x1_cpu"):
                    raise RuntimeError

                with Pool(
                        njobs=self.number_of_cores,
                        loglike=fitness,
                        prior_transform=prior_transform,
                        logl_args=(model, fitness),
                        ptform_args=(model,),
                ) as pool:

                    sampler = self.sampler_from(
                        model=model,
                        fitness=fitness,
                        checkpoint_exists=checkpoint_exists,
                        pool=pool,
                        queue_size=self.number_of_cores,
                    )

                    finished = self.run_sampler(sampler=sampler)

            except RuntimeError:

                checkpoint_exists = os.path.exists(self.checkpoint_file)

                if not checkpoint_exists:
                    self.logger.info(
                        """
                        Your operating system does not support Python multiprocessing.

                        A single CPU non-multiprocessing Dynesty run is being performed.
                        """
                    )

                sampler = self.sampler_from(
                    model=model,
                    fitness=fitness,
                    checkpoint_exists=checkpoint_exists,
                    pool=None,
                    queue_size=None,
                )

                finished = self.run_sampler(sampler=sampler)

            self.perform_update(model=model, analysis=analysis, during_analysis=True)

        self.paths.save_results_internal(obj=sampler.results)

    @property
    def samples_info(self):

        results_internal = self.sampler.results

        return {
            "log_evidence": np.max(results_internal.logz),
            "total_samples": int(np.sum(results_internal.ncall)),
            "time": self.timer.time,
            "number_live_points": self.number_live_points
        }

    def samples_via_internal_from(self, model):
        """
        Returns a `Samples` object from the dynesty internal results. 
        
        The samples contain all information on the parameter space sampling (e.g. the parameters, 
        log likelihoods, etc.).

        The internal search results are converted from the native format used by the search to lists of values
        (e.g. `parameter_lists`, `log_likelihood_list`).

        Parameters
        ----------
        model
            Maps input vectors of unit parameter values to physical values and model instances via priors.
        """
        results_internal = self.sampler.results

        parameter_lists = results_internal.samples.tolist()
        log_prior_list = model.log_prior_list_from(parameter_lists=parameter_lists)
        log_likelihood_list = list(results_internal.logl)

        try:
            weight_list = list(
                np.exp(np.asarray(results_internal.logwt) - results_internal.logz[-1])
            )
        except:
            weight_list = results_internal["weights"]

        sample_list = Sample.from_lists(
            model=model,
            parameter_lists=parameter_lists,
            log_likelihood_list=log_likelihood_list,
            log_prior_list=log_prior_list,
            weight_list=weight_list,
        )

        return SamplesNest(
            model=model,
            sample_list=sample_list,
            samples_info=self.samples_info,
            results_internal=results_internal,
        )

    @property
    def sampler(self):
        raise NotImplementedError

    def iterations_from(
            self, sampler: Union[NestedSampler, DynamicNestedSampler]
    ) -> Tuple[int, int]:
        """
        Returns the next number of iterations that a dynesty call will use and the total number of iterations
        that have been performed so far.

        This is used so that the `iterations_per_update` input leads to on-the-fly output of dynesty results.

        It also ensures dynesty does not perform more samples than the `maxcall` input variable.

        Parameters
        ----------
        sampler
            The Dynesty sampler (static or dynamic) which is run and performs nested sampling.

        Returns
        -------
        The next number of iterations that a dynesty run sampling will perform and the total number of iterations
        it has performed so far.
        """
        try:
            total_iterations = np.sum(sampler.results.ncall)
        except AttributeError:
            total_iterations = 0

        if self.config_dict_run.get("maxcall") is not None:
            iterations = self.config_dict_run["maxcall"] - total_iterations

            return int(iterations), int(total_iterations)
        return self.iterations_per_update, int(total_iterations)

    def run_sampler(self, sampler: Union[NestedSampler, DynamicNestedSampler]):
        """
        Run the Dynesty sampler, which could be either the static of dynamic sampler.

        The number of iterations which the sampler runs from depends on the `maxcall` input. Due to on-to-fly updates
        via `perform_update` the number of remaining iterations compared to `maxcall` is tracked between every
        `run_nested` call, and whether or not the sampler is finished is returned at the end of this function.

        A second finish criteria is used, which occurs when the dynesty run performs zero likelihood updates (because
        the sampling accrording to Dynesty's termination criteria is complete).

        Parameters
        ----------
        sampler
            The Dynesty sampler (static or dynamic) which is run and performs nested sampling.

        Returns
        -------

        """

        iterations, total_iterations = self.iterations_from(sampler=sampler)

        config_dict_run = {key: value for key, value in self.config_dict_run.items() if key != 'maxcall'}

        if iterations > 0:
            sampler.run_nested(
                maxcall=iterations,
                print_progress=not self.silence,
                checkpoint_file=self.checkpoint_file,
                **config_dict_run,
            )

        iterations_after_run = np.sum(sampler.results.ncall)

        return (
                total_iterations == iterations_after_run
                or total_iterations == self.config_dict_run.get("maxcall")
        )

    def write_uses_pool(self, uses_pool: bool) -> str:
        """
        If a Dynesty fit does not use a parallel pool, and is then resumed using one,
        this causes significant slow down.

        This file checks the original pool use so an exception can be raised to avoid this.
        """
        with open(self.paths.search_internal_path / "uses_pool.save", "w+") as f:
            if uses_pool:
                f.write("True")
            else:
                f.write("")

    def read_uses_pool(self) -> str:
        """
        If a Dynesty fit does not use a parallel pool, and is then resumed using one,
        this causes significant slow down.

        This file checks the original pool use so an exception can be raised to avoid this.
        """
        with open(self.paths.search_internal_path / "uses_pool.save", "r+") as f:
            return bool(f.read())

    @property
    def checkpoint_file(self) -> str:
        """
        The path to the file used by dynesty for checkpointing.
        """
        return str(self.paths.search_internal_path / "savestate.save")

    def config_dict_with_test_mode_settings_from(self, config_dict):

        return {
            **config_dict,
            "maxiter": 1,
            "maxcall": 1,
        }

    def live_points_init_from(self, model, fitness):
        """
        By default, dynesty live points are generated via the sampler's in-built initialization.

        However, in test-mode this would take a long time to run, thus we overwrite the initial live points
        with quickly generated samplers from the initializer.

        Parameters
        ----------
        model
        fitness

        Returns
        -------

        """

        (
            unit_parameters,
            parameters,
            log_likelihood_list,
        ) = self.initializer.samples_from_model(
            total_points=self.number_live_points,
            model=model,
            fitness=fitness,
        )

        init_unit_parameters = np.zeros(
            shape=(self.number_live_points, model.prior_count)
        )
        init_parameters = np.zeros(
            shape=(self.number_live_points, model.prior_count)
        )
        init_log_likelihood_list = np.zeros(shape=(self.number_live_points))

        for i in range(len(parameters)):
            init_unit_parameters[i, :] = np.asarray(unit_parameters[i])
            init_parameters[i, :] = np.asarray(parameters[i])
            init_log_likelihood_list[i] = np.asarray(log_likelihood_list[i])

        live_points = [
            init_unit_parameters,
            init_parameters,
            init_log_likelihood_list,
        ]

        blobs = np.asarray(self.number_live_points * [False])

        live_points.append(blobs)

        return live_points

    def sampler_from(
            self,
            model: AbstractPriorModel,
            fitness,
            checkpoint_exists: bool,
            pool: Optional,
            queue_size: Optional[int],
    ):
        raise NotImplementedError()

    def check_pool(self, uses_pool: bool, pool):

        if (uses_pool and pool is None) or (not uses_pool and pool is not None):
            raise exc.SearchException(
                """
                A Dynesty sampler has been loaded and its pool type is not the same as the input pool type.

                This means that the original samples in dynesty were computed with or without a 
                multiprocessing pool, whereas the run is now trying to use a multiprocessing pool.

                This could indiciate the number of cores have change values or Python multiprocessing
                has been disabled and then enabled.
                """
            )

    def remove_state_files(self):

        os.remove(self.checkpoint_file)

    @property
    def number_live_points(self):
        raise NotImplementedError()

    def plot_results(self, samples):

        from autofit.non_linear.search.nest.dynesty.plotter import DynestyPlotter

        if not samples.pdf_converged:
            return

        def should_plot(name):
            return conf.instance["visualize"]["plots_search"]["dynesty"][name]

        plotter = DynestyPlotter(
            samples=samples,
            output=Output(
                path=self.paths.image_path / "search", format="png"
            ),
        )

        if should_plot("cornerplot"):
            plotter.cornerplot()

        if should_plot("traceplot"):
            plotter.traceplot()

        # There is currently a bug internal in dynesty where the matplotlib figure produced after these plots
        # is not closed, and has weird extra stuff on. I have commented these out for now, in the hope that dynesty
        # fix this bug in the future.

        # if should_plot("runplot"):
        #     plotter.runplot()

        # if should_plot("cornerpoints"):
        #     plotter.cornerpoints()