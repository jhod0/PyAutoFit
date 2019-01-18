import pytest

from autofit import mock
from autofit.mapper import model_mapper as mm
from autofit.optimize import grid_search as gs
from autofit.optimize import non_linear


@pytest.fixture(name="mapper")
def make_mapper():
    mapper = mm.ModelMapper()
    mapper.profile = mock.GeometryProfile
    return mapper


@pytest.fixture(name="grid_search")
def make_grid_search(mapper):
    return gs.GridSearch(model_mapper=mapper, step_size=0.1)


class TestGridSearchablePriors(object):
    def test_generated_models(self, grid_search):
        mappers = list(grid_search.models_mappers(
            grid_priors=[grid_search.variable.profile.centre_0, grid_search.variable.profile.centre_1]))

        assert len(mappers) == 100

        assert mappers[0].profile.centre_0.lower_limit == 0.0
        assert mappers[0].profile.centre_0.upper_limit == 0.1
        assert mappers[0].profile.centre_1.lower_limit == 0.0
        assert mappers[0].profile.centre_1.upper_limit == 0.1

        assert mappers[-1].profile.centre_0.lower_limit == 0.9
        assert mappers[-1].profile.centre_0.upper_limit == 1.0
        assert mappers[-1].profile.centre_1.lower_limit == 0.9
        assert mappers[-1].profile.centre_1.upper_limit == 1.0

    def test_non_grid_searched_dimensions(self, mapper):
        grid_search = gs.GridSearch(model_mapper=mapper, step_size=0.1)
        mappers = list(grid_search.models_mappers(grid_priors=[mapper.profile.centre_0]))

        assert len(mappers) == 10

        assert mappers[0].profile.centre_0.lower_limit == 0.0
        assert mappers[0].profile.centre_0.upper_limit == 0.1
        assert mappers[0].profile.centre_1.lower_limit == 0.0
        assert mappers[0].profile.centre_1.upper_limit == 1.0

        assert mappers[-1].profile.centre_0.lower_limit == 0.9
        assert mappers[-1].profile.centre_0.upper_limit == 1.0
        assert mappers[-1].profile.centre_1.lower_limit == 0.0
        assert mappers[-1].profile.centre_1.upper_limit == 1.0

    def test_tied_priors(self, grid_search):
        grid_search.variable.profile.centre_0 = grid_search.variable.profile.centre_1

        mappers = list(grid_search.models_mappers(
            grid_priors=[grid_search.variable.profile.centre_0, grid_search.variable.profile.centre_1]))

        assert len(mappers) == 10

        assert mappers[0].profile.centre_0.lower_limit == 0.0
        assert mappers[0].profile.centre_0.upper_limit == 0.1
        assert mappers[0].profile.centre_1.lower_limit == 0.0
        assert mappers[0].profile.centre_1.upper_limit == 0.1

        assert mappers[-1].profile.centre_0.lower_limit == 0.9
        assert mappers[-1].profile.centre_0.upper_limit == 1.0
        assert mappers[-1].profile.centre_1.lower_limit == 0.9
        assert mappers[-1].profile.centre_1.upper_limit == 1.0

        for mapper in mappers:
            assert mapper.profile.centre_0 == mapper.profile.centre_1


class TestGridNLOBehaviour(object):
    def test_calls(self, mapper):
        init_args = []
        fit_args = []

        class MockOptimizer(non_linear.NonLinearOptimizer):
            def __init__(self, model_mapper, name):
                super().__init__(model_mapper, name)
                init_args.append((model_mapper, name))

            def fit(self, analysis):
                fit_args.append(analysis)

        class MockAnalysis(non_linear.Analysis):
            def fit(self, instance):
                return 1

            def visualize(self, instance, suffix, during_analysis):
                pass

            def log(self, instance):
                pass

        grid_search = gs.GridSearch(model_mapper=mapper, optimizer_class=MockOptimizer, step_size=0.1)

        results = grid_search.fit(MockAnalysis(), [mapper.profile.centre_0])

        assert len(init_args) == 10
        assert len(fit_args) == 10
        assert len(results) == 10
