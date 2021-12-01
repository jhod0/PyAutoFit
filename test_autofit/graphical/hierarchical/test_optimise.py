import pytest

import autofit as af


@pytest.fixture(
    name="factor"
)
def make_factor(
        hierarchical_factor
):
    return hierarchical_factor.factors[0]


def test_optimise(factor):
    optimizer = af.DynestyStatic(
        maxcall=10
    )

    _, status = optimizer.optimise(
        factor,
        factor.mean_field_approximation()
    )
    assert status


def test_instance(factor):
    prior_model = factor.prior_model
    assert len(prior_model.priors) == 3
    _ = factor.prior_model.instance_from_prior_medians()
