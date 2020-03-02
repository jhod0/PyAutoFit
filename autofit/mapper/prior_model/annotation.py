from autofit.mapper.prior_model.assertion import (
    GreaterThanLessThanAssertion,
    GreaterThanLessThanEqualAssertion,
)
from autofit.mapper.prior_model.prior_model import PriorModel, Prior


class AnnotationPriorModel(PriorModel):
    def __init__(self, cls, parent_class, true_argument_name, **kwargs):
        self.parent_class = parent_class
        self.true_argument_name = true_argument_name
        self._value = None
        super().__init__(cls, **kwargs)

    def make_prior(self, attribute_name):
        if self._value is None:
            self._value = Prior.for_class_and_attribute_name(
                self.parent_class, self.true_argument_name
            )
        return self._value

    def __gt__(self, other_prior: "Prior") -> GreaterThanLessThanAssertion:
        """
        Add an assertion that values associated with this prior are greater.

        Parameters
        ----------
        other_prior
            Another prior which is associated with a field that should always have
            lower physical values.

        Returns
        -------
        An assertion object
        """
        return GreaterThanLessThanAssertion(greater=self, lower=other_prior)

    def __lt__(self, other_prior: "Prior") -> GreaterThanLessThanAssertion:
        """
        Add an assertion that values associated with this prior are lower.

        Parameters
        ----------
        other_prior
            Another prior which is associated with a field that should always have
            greater physical values.

        Returns
        -------
        An assertion object
        """
        return GreaterThanLessThanAssertion(lower=self, greater=other_prior)

    def __ge__(self, other_prior: "Prior") -> GreaterThanLessThanEqualAssertion:
        """
        Add an assertion that values associated with this prior are greater or equal.

        Parameters
        ----------
        other_prior
            Another prior which is associated with a field that should always have
            lower physical values.

        Returns
        -------
        An assertion object
        """
        return GreaterThanLessThanEqualAssertion(greater=self, lower=other_prior)

    def __le__(self, other_prior: "Prior") -> GreaterThanLessThanEqualAssertion:
        """
        Add an assertion that values associated with this prior are lower or equal.

        Parameters
        ----------
        other_prior
            Another prior which is associated with a field that should always have
            greater physical values.

        Returns
        -------
        An assertion object
        """
        return GreaterThanLessThanEqualAssertion(lower=self, greater=other_prior)
