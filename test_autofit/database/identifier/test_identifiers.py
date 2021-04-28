import autofit as af
from autofit import database as db
from autofit.mapper.model_object import Identifier
from autofit.mock.mock import Gaussian


class Class:
    def __init__(self, one=1, two=2, three=3):
        self.one = one
        self.two = two
        self.three = three

    __identifier_fields__ = ("one", "two")


def test_identifier_fields():
    other = Class(three=4)
    assert Identifier(
        Class()
    ) == Identifier(
        other
    )

    other.__identifier_fields__ = ("one", "two", "three")
    assert Identifier(
        Class()
    ) != Identifier(
        other
    )


def test_fit():
    assert db.Fit(
        info={"info": 1}
    ).id != db.Fit(
        info={"info": 2}
    ).id


def test_prior():
    identifier = af.UniformPrior().identifier
    assert identifier == af.UniformPrior().identifier
    assert identifier != af.UniformPrior(
        lower_limit=0.5
    ).identifier
    assert identifier != af.UniformPrior(
        upper_limit=0.5
    ).identifier


def test_model():
    identifier = af.PriorModel(
        Gaussian,
        centre=af.UniformPrior()
    ).identifier
    assert identifier == af.PriorModel(
        Gaussian,
        centre=af.UniformPrior()
    ).identifier
    assert identifier != af.PriorModel(
        Gaussian,
        centre=af.UniformPrior(
            upper_limit=0.5
        )
    ).identifier


def test_collection():
    identifier = af.CollectionPriorModel(
        gaussian=af.PriorModel(
            Gaussian,
            centre=af.UniformPrior()
        )
    ).identifier
    assert identifier == af.CollectionPriorModel(
        gaussian=af.PriorModel(
            Gaussian,
            centre=af.UniformPrior()
        )
    ).identifier
    assert identifier != af.CollectionPriorModel(
        gaussian=af.PriorModel(
            Gaussian,
            centre=af.UniformPrior(
                upper_limit=0.5
            )
        )
    ).identifier


def test_instance():
    identifier = af.CollectionPriorModel(
        gaussian=Gaussian()
    ).identifier
    assert identifier == af.CollectionPriorModel(
        gaussian=Gaussian()
    ).identifier
    assert identifier != af.CollectionPriorModel(
        gaussian=Gaussian(
            centre=0.5
        )
    ).identifier
