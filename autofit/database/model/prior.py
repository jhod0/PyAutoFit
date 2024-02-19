import dill
from typing import Union

from ..sqlalchemy_ import sa

from autofit.mapper.prior import abstract
from autofit.mapper.prior_model import prior_model
from autofit.mapper.prior_model import collection

from .model import Object


class Collection(Object):
    """
    A collection
    """

    __tablename__ = "collection_prior_model"

    id = sa.Column(
        sa.Integer,
        sa.ForeignKey("object.id"),
        primary_key=True,
        index=True,
    )

    __mapper_args__ = {"polymorphic_identity": "collection_prior_model"}

    @classmethod
    def _from_object(cls, source: Union[collection.Collection, list, dict]):
        instance = cls()
        if not isinstance(source, collection.Collection):
            source = collection.Collection(source)
        instance._add_children(source.items())
        instance.cls = collection.Collection
        return instance


class Model(Object):
    """
    A prior model
    """

    __tablename__ = "prior_model"

    id = sa.Column(
        sa.Integer,
        sa.ForeignKey("object.id"),
        primary_key=True,
        index=True,
    )

    __mapper_args__ = {"polymorphic_identity": "prior_model"}

    @classmethod
    def _from_object(
        cls,
        model: prior_model.Model,
    ):
        instance = cls()
        instance.cls = model.cls
        instance._add_children(
            model.items()
            + [
                (f"assertion_{i}", assertion)
                for i, assertion in enumerate(model.assertions)
            ]
            + [
                (f"custom_derived_quantity_{key}", value)
                for key, value in model.custom_derived_quantities.items()
            ]
        )
        return instance

    def _make_instance(self):
        instance = object.__new__(prior_model.Model)
        instance.cls = self.cls
        instance._assertions = []

        instance._custom_derived_quantities = {}
        return instance


class Prior(Object):
    """
    A prior
    """

    __tablename__ = "prior"

    id = sa.Column(
        sa.Integer,
        sa.ForeignKey("object.id"),
        primary_key=True,
        index=True,
    )

    __mapper_args__ = {"polymorphic_identity": "prior"}

    @classmethod
    def _from_object(cls, model: abstract.Prior):
        instance = cls()
        instance.cls = type(model)
        instance._add_children(
            [(key, getattr(model, key)) for key in model.__database_args__]
        )
        return instance

    def __call__(self):
        """
        Create the real instance for this object, with child
        attributes attached.

        If the instance implements __setstate__ then this is
        called with a dictionary of instantiated children.
        """
        arguments = {child.name: child() for child in self.children}
        return self.cls(**arguments)
