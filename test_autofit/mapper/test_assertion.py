import pytest

import autofit as af


@pytest.fixture(
    name="prior_1"
)
def make_prior_1():
    return af.UniformPrior()


@pytest.fixture(
    name="prior_2"
)
def make_prior_2():
    return af.UniformPrior()


@pytest.fixture(
    name="lower_assertion"
)
def make_lower_assertion(
        prior_1,
        prior_2
):
    return prior_1 < prior_2


@pytest.fixture(
    name="greater_assertion"
)
def make_greater_assertion(
        prior_1,
        prior_2
):
    return prior_1 > prior_2


class TestAssertion:
    def test_lower_assertion(
            self,
            lower_assertion,
            prior_1,
            prior_2
    ):
        assert isinstance(
            lower_assertion,
            af.Assertion
        )

        assert lower_assertion.lower is prior_1
        assert lower_assertion.greater is prior_2

    def test_greater_assertion(
            self,
            greater_assertion,
            prior_1,
            prior_2
    ):
        assert isinstance(
            greater_assertion,
            af.Assertion
        )

        assert greater_assertion.lower is prior_2
        assert greater_assertion.greater is prior_1

    def test_assert_on_arguments_lower(
            self,
            lower_assertion,
            prior_1,
            prior_2
    ):
        lower_assertion(
            {
                prior_1: 0.3,
                prior_2: 0.5
            }
        )
        with pytest.raises(af.exc.FitException):
            lower_assertion(
                {
                    prior_1: 0.6,
                    prior_2: 0.5
                }
            )

    def test_assert_on_arguments_greater(
            self,
            greater_assertion,
            prior_1,
            prior_2
    ):
        greater_assertion(
            {
                prior_1: 0.6,
                prior_2: 0.5
            }
        )
        with pytest.raises(af.exc.FitException):
            greater_assertion(
                {
                    prior_1: 0.3,
                    prior_2: 0.5
                }
            )


def test_assertion_in_model(
        prior_1,
        prior_2
):
    model = af.ModelMapper()
    model.one = prior_1
    model.two = prior_2

    model.add_assertion(
        prior_1 < prior_2
    )

    model.instance_from_unit_vector(
        [0.1, 0.2]
    )
    with pytest.raises(
            af.exc.FitException
    ):
        model.instance_from_unit_vector(
            [0.2, 0.1]
        )