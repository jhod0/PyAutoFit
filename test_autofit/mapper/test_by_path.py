import pytest

import autofit as af
from autofit.mock.mock import WithTuple


@pytest.fixture(
    name="model"
)
def make_model():
    return af.Collection(
        gaussian=af.Model(
            af.Gaussian
        )
    )


class TestInstanceFromPathArguments:
    def test(
            self,
            model
    ):
        instance = model.instance_from_path_arguments({
            ("gaussian", "centre"): 0.1,
            ("gaussian", "intensity"): 0.2,
            ("gaussian", "sigma"): 0.3
        })
        assert instance.gaussian.centre == 0.1
        assert instance.gaussian.intensity == 0.2
        assert instance.gaussian.sigma == 0.3

    def test_prior_linking(
            self,
            model
    ):
        model.gaussian.centre = model.gaussian.intensity
        instance = model.instance_from_path_arguments({
            ("gaussian", "centre",): 0.1,
            ("gaussian", "sigma",): 0.3
        })
        assert instance.gaussian.centre == 0.1
        assert instance.gaussian.intensity == 0.1
        assert instance.gaussian.sigma == 0.3

        instance = model.instance_from_path_arguments({
            ("gaussian", "intensity",): 0.2,
            ("gaussian", "sigma",): 0.3
        })
        assert instance.gaussian.centre == 0.2
        assert instance.gaussian.intensity == 0.2
        assert instance.gaussian.sigma == 0.3


@pytest.fixture(
    name="underscore_model"
)
def make_underscore_model():
    return af.Collection(
        gaussian_component=af.Model(
            af.Gaussian
        )
    )


class TestInstanceFromPriorNames:
    def test(self, model):
        instance = model.instance_from_prior_name_arguments({
            "gaussian_centre": 0.1,
            "gaussian_intensity": 0.2,
            "gaussian_sigma": 0.3
        })
        assert instance.gaussian.centre == 0.1
        assert instance.gaussian.intensity == 0.2
        assert instance.gaussian.sigma == 0.3

    def test_underscored_names(self, underscore_model):
        instance = underscore_model.instance_from_prior_name_arguments({
            "gaussian_component_centre": 0.1,
            "gaussian_component_intensity": 0.2,
            "gaussian_component_sigma": 0.3
        })
        assert instance.gaussian_component.centre == 0.1
        assert instance.gaussian_component.intensity == 0.2
        assert instance.gaussian_component.sigma == 0.3

    def test_prior_linking(self, underscore_model):
        underscore_model.gaussian_component.intensity = (
            underscore_model.gaussian_component.centre
        )
        instance = underscore_model.instance_from_prior_name_arguments({
            "gaussian_component_centre": 0.1,
            "gaussian_component_sigma": 0.3
        })
        assert instance.gaussian_component.centre == 0.1
        assert instance.gaussian_component.intensity == 0.1
        assert instance.gaussian_component.sigma == 0.3

        instance = underscore_model.instance_from_prior_name_arguments({
            "gaussian_component_intensity": 0.2,
            "gaussian_component_sigma": 0.3
        })
        assert instance.gaussian_component.centre == 0.2
        assert instance.gaussian_component.intensity == 0.2
        assert instance.gaussian_component.sigma == 0.3

    def test_path_for_name(self, underscore_model):
        assert underscore_model.path_for_name(
            "gaussian_component_centre"
        ) == (
                   "gaussian_component",
                   "centre"
               )


def test_component_names():
    model = af.Model(
        af.Gaussian
    )
    assert model.model_component_and_parameter_names == [
        'centre', 'intensity', 'sigma'
    ]


def test_with_tuple():
    with_tuple = af.Model(
        WithTuple
    )
    assert with_tuple.model_component_and_parameter_names == [
        "tup_0", "tup_1"
    ]


@pytest.fixture(
    name="linked_model"
)
def make_linked_model():
    model = af.Model(
        af.Gaussian
    )
    model.sigma = model.centre
    return model


class TestAllPaths:
    def test_independent(self):
        model = af.Model(
            af.Gaussian
        )

        assert model.all_paths_prior_tuples == [
            ((("centre",),), model.centre),
            ((("intensity",),), model.intensity),
            ((("sigma",),), model.sigma),
        ]

    def test_linked(self, linked_model):
        assert linked_model.all_paths_prior_tuples == [
            ((("centre",), ("sigma",)), linked_model.centre),
            ((("intensity",),), linked_model.intensity)
        ]

    def test_names_independent(self):
        model = af.Model(
            af.Gaussian
        )

        assert model.all_name_prior_tuples == [
            (("centre",), model.centre),
            (("intensity",), model.intensity),
            (("sigma",), model.sigma),
        ]

    def test_names_linked(self, linked_model):
        assert linked_model.all_name_prior_tuples == [
            (("centre", "sigma"), linked_model.centre),
            (("intensity",), linked_model.intensity)
        ]


def test_changing_model(model):
    samples = af.OptimizerSamples(
        model,
        [
            af.Sample(
                log_likelihood=1.0,
                log_prior=1.0,
                weight=1.0,
                kwargs={
                    ("gaussian", "centre"): 0.1,
                    ("gaussian", "intensity"): 0.2,
                    ("gaussian", "sigma"): 0.3,
                }
            )
        ]
    )

    result = af.Result(
        samples,
        model
    )

    instance = result.max_log_likelihood_instance

    assert instance.gaussian.centre == 0.1
    assert instance.gaussian.intensity == 0.2
    assert instance.gaussian.sigma == 0.3
