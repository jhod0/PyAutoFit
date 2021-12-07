import autofit as af
from autoconf.conf import with_config

MAX_STEPS = 3


class MockResult(af.MockResult):
    def __init__(self, model):
        super().__init__()
        self.model = model

    @property
    def projected_model(self):
        return self.model


class MockSearch(af.MockSearch):
    def fit(
            self,
            model,
            analysis,
            info=None,
            pickle_files=None,
            log_likelihood_cap=None
    ):
        super().fit(
            model,
            analysis,
        )
        return MockResult(model)


def _run_optimisation(
        factor_graph_model
):
    factor_graph_model.optimise(
        MockSearch(),
        max_steps=MAX_STEPS,
        name="name",
        log_interval=1,
        visualise_interval=1,
        output_interval=1,
    )


@with_config(
    "general",
    "output",
    "remove_files",
    value=False
)
def test_output(
        output_directory,
        factor_graph_model
):
    factor_graph_model.model_factors[0]._name = "factor_1"
    factor_graph_model.model_factors[1]._name = "factor_2"
    _run_optimisation(factor_graph_model)

    path = output_directory / "name/factor_1"

    assert path.exists()
    assert (output_directory / "name/factor_2").exists()

    for number in range(MAX_STEPS):
        assert (path / f"optimization_{number}").exists()


@with_config(
    "general",
    "output",
    "remove_files",
    value=False
)
def test_default_output(
        output_directory,
        factor_graph_model
):
    _run_optimisation(factor_graph_model)
    assert (output_directory / "name/AnalysisFactor0").exists()
    assert (output_directory / "name/AnalysisFactor1").exists()