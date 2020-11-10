from abc import ABC
from inspect import getfullargspec
from itertools import chain, repeat
from typing import \
    Tuple, Dict, Union, Set, NamedTuple, Callable, Optional
from functools import lru_cache
import numpy as np


from autofit.graphical.utils import aggregate, Axis
from autofit.graphical.factor_graphs.abstract import \
    AbstractNode, FactorValue, JacobianValue
from autofit.mapper.variable import Variable


class AbstractFactor(AbstractNode, ABC):
    def __init__(
            self,
            name=None,
            **kwargs: Variable,
    ):
        super().__init__(**kwargs)
        self._name = name or f"factor_{self.id}"
        self._deterministic_variables = set()

    @property
    def deterministic_variables(self) -> Set[Variable]:
        """
        Dictionary mapping the names of deterministic variables to those variables
        """
        return self._deterministic_variables

    @property
    def name(self) -> str:
        return self._name

    def __mul__(self, other):
        """
        When two factors are multiplied together this creates a graph
        """
        from autofit.graphical.factor_graphs.graph import FactorGraph
        return FactorGraph([self]) * other

    @property
    def variables(self) -> Set[Variable]:
        """
        Dictionary mapping the names of variables to those variables
        """
        return set(self._kwargs.values())

    @property
    def _kwargs_dims(self) -> Dict[str, int]:
        """
        The number of plates for each keyword argument variable
        """
        return {
            key: len(value)
            for key, value
            in self._kwargs.items()
        }

    @property
    def _variable_plates(self) -> Dict[str, np.ndarray]:
        """
        Maps the name of each variable to the indices of its plates
        within this node
        """
        return {
            variable: self._match_plates(
                variable.plates
            )
            for variable
            in self.all_variables
        }

    @property
    def n_deterministic(self) -> int:
        """
        How many deterministic variables are there associated with this node?
        """
        return len(self._deterministic_variables)

    def __hash__(self):
        return hash((type(self), self.id))

    def _resolve_args(
            self,
            **kwargs: np.ndarray
    ) -> dict:
        """
        Transforms in the input arguments to match the arguments
        specified for the factor.

        Parameters
        ----------
        args
        kwargs

        Returns
        -------

        """
        return {n: kwargs[v.name] for n, v in self._kwargs.items()}
    

class Factor(AbstractFactor):
    def __init__(
            self,
            factor: Callable,
            name=None,
            vectorised=False,
            **kwargs: Variable
    ):
        """
        A node in a graph representing a factor

        Parameters
        ----------
        factor
            A wrapper around some callable
        args
            Variables representing positional arguments for the function
        kwargs
            Variables representing keyword arguments for the function
        """
        self.vectorised = vectorised

        self._factor = factor

        args = getfullargspec(self._factor).args
        kwargs = {
            **kwargs,
            **{
                arg: Variable(arg)
                for arg
                in args
                if arg not in kwargs and arg != "self"
            }
        }

        super().__init__(
            **kwargs,
            name=name or factor.__name__
        )

    # jacobian = numerical_jacobian

    def __hash__(self) -> int:
        # TODO: might this break factor repetition somewhere?
        return hash(self._factor)

    def _function_shape(
            self, 
            **kwargs: np.ndarray) -> Tuple[int, ...]:
        """
        Calculates the expected function shape based on the variables
        """
        var_shapes = {
            k: np.shape(x) for k, x in kwargs.items()}
        return self._var_shape(**var_shapes)
    
    @lru_cache(maxsize=8)
    def _var_shape(self, **kwargs: Tuple[int, ...]) -> Tuple[int, ...]:
        """This is called by _function_shape
        
        caches result so that does not have to be recalculated each call
        
        lru_cache caches f(x=1, y=2) to f(y=2, x=1), but in this case
        it should be find as the order of kwargs is set by self._kwargs
        which should be stable
        """
        var_shapes = {self._kwargs[k]: v for k, v in kwargs.items()}
        var_dims_diffs = {
            v: len(s) - v.ndim
            for v, s in var_shapes.items()
        }
        """
        If all the passed variables have an extra dimension then 
        we assume we're evaluating multiple instances of the function at the 
        same time

        otherwise an error is raised
        """
        if set(var_dims_diffs.values()) == {1}:
            # Check if we're passing multiple values e.g. for sampling
            shift = 1
        elif set(var_dims_diffs.values()) == {0}:
            shift = 0
        else:
            raise ValueError("dimensions of passed inputs do not match")

        """
        Updating shape of output array to match input arrays

        singleton dimensions are always assumed to match as in
        standard array broadcasting

        e.g. (1, 2, 3) == (3, 2, 1)
        """
        shape = np.ones(self.ndim + shift, dtype=int)
        for v, vs in var_shapes.items():
            ind = self._variable_plates[v] + shift
            vshape = vs[shift:]
            if shift:
                ind = np.r_[0, ind]
                vshape = (vs[0],) + vshape

            if shape.size:
                if not (
                        np.equal(shape[ind], 1) |
                        np.equal(shape[ind], vshape) |
                        np.equal(vshape, 1)).all():
                    raise AssertionError(
                        "Shapes do not match"
                    )
                shape[ind] = np.maximum(shape[ind], vshape)
        
        return tuple(shape)

    def _call_factor(
            self,
            **kwargs: np.ndarray
    ) -> np.ndarray:
        """
        Call the underlying function

        Parameters
        ----------
        args
            Positional arguments for the function
        kwargs
            Keyword arguments for the function

        Returns
        -------
        Value returned by the factor
        """
        # kws = self._resolve_args(
        #     **kwargs
        # )

        if self.vectorised:
            return self._factor(**kwargs)
        return self._py_vec_call(**kwargs)

    def _py_vec_call(
            self,
            **kwargs: np.ndarray
    ) -> np.ndarray:
        """Some factors may not be vectorised to broadcast over
        multiple inputs

        this method checks whether multiple input values have been
        passed, and if so automatically loops over the inputs.
        If any of the inputs have initial dimension one, it repeats
        that value to match the length of the other inputs

        If the other inputs do not match then it raises ValueError
        """
        kwargs_dims = {k: np.ndim(a) for k, a in kwargs.items()}
        # Check dimensions of inputs directly match plates
        direct_call = (
            all(dim == kwargs_dims[k] for k, dim in self._kwargs_dims.items()))
        if direct_call:
            return self._factor(**kwargs)

        # Check dimensions of inputs match plates + 1
        vectorised = (
            all(dim + 1 == kwargs_dims[k]
                for k, dim in self._kwargs_dims.items()))

        if not vectorised:
            raise ValueError(
                "input dimensions do not match required dims"
                f"input: **kwargs={kwargs_dims}"
                f"required: "
                f"**kwargs={self._kwargs_dims}")

        kw_lens = {k: len(a) for k, a in kwargs.items()}

        # checking 1st dimensions match
        sizes = set(kw_lens.values())
        dim0 = max(sizes)
        if sizes.difference({1, dim0}):
            raise ValueError(
                f"size mismatch first dimensions passed: {sizes}")

        iter_kws = {
            k: iter(a) if kw_lens[k] == dim0 else iter(repeat(a[0]))
            for k, a in kwargs.items()}

        # iterator to generate keyword arguments
        def gen_kwargs():
            for _ in range(dim0):
                yield {
                    k: next(a) for k, a in iter_kws.items()}

        # TODO this loop can also be parallelised for increased performance
        res = np.array([
            self._factor(**kws)
            for kws in gen_kwargs()])

        return res

    # @accept_variable_dict
    def __call__(
            self,
            variable_dict: Dict[Variable, np.ndarray],
            axis: Axis = False, 
            # **kwargs: np.ndarray
    ) -> FactorValue:
        """
        Call the underlying factor

        Parameters
        ----------
        args
            Positional arguments for the factor
        kwargs
            Keyword arguments for the factor

        Returns
        -------
        Object encapsulating the result of the function call
        """
        kwargs = self.resolve_variable_dict(variable_dict)
        val = self._call_factor(**kwargs)
        val = aggregate(
            val.reshape(self._function_shape(**kwargs)), axis)
        return FactorValue(val, {})

    def broadcast_variable(
            self,
            variable: str,
            value: np.ndarray
    ) -> np.ndarray:
        """
        broadcasts the value of a variable to match the specific shape
        of the factor

        if the number of dimensions passed of the variable is 1
        greater than the dimensions of the variable then it's assumed
        that that dimension corresponds to multiple samples of that variable
        """
        return self._broadcast(
            self._variable_plates[variable],
            value
        )

    def collapse(
            self,
            variable: str,
            value: np.ndarray,
            agg_func=np.sum
    ) -> np.ndarray:
        """
        broadcasts the value of a variable to match the specific shape
        of the factor

        if the number of dimensions passed of the variable is 1
        greater than the dimensions of the variable then it's assumed
        that that dimension corresponds to multiple samples of that variable
        """
        ndim = np.ndim(value)
        shift = ndim - self.ndim
        assert shift in {0, 1}
        inds = self._variable_plates[variable] + shift
        dropaxes = tuple(np.setdiff1d(
            np.arange(shift, ndim), inds))

        # to ensured axes of returned array is in the correct order
        moved = np.moveaxis(value, inds, np.sort(inds))
        return agg_func(moved, axis=dropaxes)

    def __eq__(self, other: Union["Factor", Variable]):
        """
        If set equal to a variable that variable is taken to be deterministic and
        so a DeterministicFactorNode is generated.
        """
        if isinstance(other, Factor):
            if isinstance(other, type(self)):
                return (
                        (self._factor == other._factor)
                        and (frozenset(self._kwargs.items())
                             == frozenset(other._kwargs.items()))
                        and (frozenset(self.variables)
                             == frozenset(other.variables))
                        and (frozenset(self.deterministic_variables)
                             == frozenset(self.deterministic_variables)))
            else:
                return False

        from autofit.graphical.factor_graphs import DeterministicFactorNode
        return DeterministicFactorNode(
            self._factor,
            other,
            **self._kwargs
        )

    def __repr__(self) -> str:
        args = ", ".join(chain(
            map("{0[0]}={0[1]}".format, self._kwargs.items())))
        return f"Factor({self.name}, {args})"



class FactorJacobian(Factor):
    def __init__(
            self,
            factor_jacobian: Callable,
            name=None,
            vectorised=False,
            variable_order=None, 
            **kwargs: Variable
    ):
        """
        A node in a graph representing a factor

        Parameters
        ----------
        factor
            A wrapper around some callable
        args
            Variables representing positional arguments for the function
        kwargs
            Variables representing keyword arguments for the function
        """
        self.vectorised = vectorised
        self._factor_jacobian = factor_jacobian
        AbstractFactor.__init__(
            self, 
            **kwargs,
            name=name or factor_jacobian.__name__
        )
        self._variables = tuple(
            self._kwargs.values() if variable_order is None else variable_order)


    def __hash__(self) -> int:
        # TODO: might this break factor repetition somewhere?
        return hash(self._factor)

    def _call_factor(
            self,
            values: Dict[str, np.ndarray],
            variables: Optional[Tuple[str, ...]] = None, 
    ) -> Union[np.ndarray, Tuple[np.ndarray, Tuple[np.ndarray, ...]]]:
        """
        Call the underlying function

        Parameters
        ----------
        args
            Positional arguments for the function
        kwargs
            Keyword arguments for the function

        Returns
        -------
        Value returned by the factor
        """
        if self.vectorised:
            return self._factor_jacobian(**values, _variables=variables)
            
        """Some factors may not be vectorised to broadcast over
        multiple inputs

        this method checks whether multiple input values have been
        passed, and if so automatically loops over the inputs.
        If any of the inputs have initial dimension one, it repeats
        that value to match the length of the other inputs

        If the other inputs do not match then it raises ValueError
        """
        kwargs_dims = {k: np.ndim(a) for k, a in values.items()}
        # Check dimensions of inputs directly match plates
        direct_call = (
            all(dim == kwargs_dims[k] for k, dim in self._kwargs_dims.items()))
        if direct_call:
            return self._factor_jacobian(**values, _variables=variables)

        # Check dimensions of inputs match plates + 1
        vectorised = (
            all(dim + 1 == kwargs_dims[k]
                for k, dim in self._kwargs_dims.items()))

        if not vectorised:
            raise ValueError(
                "input dimensions do not match required dims"
                f"input: **kwargs={kwargs_dims}"
                f"required: "
                f"**kwargs={self._kwargs_dims}")

        kw_lens = {k: len(a) for k, a in values.items()}

        # checking 1st dimensions match
        sizes = set(kw_lens.values())
        dim0 = max(sizes)
        if sizes.difference({1, dim0}):
            raise ValueError(
                f"size mismatch first dimensions passed: {sizes}")

        iter_kws = {
            k: iter(a) if kw_lens[k] == dim0 else iter(repeat(a[0]))
            for k, a in values.items()}

        # iterator to generate keyword arguments
        def gen_kwargs():
            for _ in range(dim0):
                yield {
                    k: next(a) for k, a in iter_kws.items()}

        # TODO this loop can also be parallelised for increased performance
        fjacs = [
            self._factor_jacobian(**kws, _variables=variables)
             for kws in gen_kwargs()]
        res = np.array([fjac[0] for fjac in fjacs])
        if variables is None:
            return res 
        else:
            njac = len(fjacs[0][1])
            jacs = tuple(
                np.array([fjac[1][i] for fjac in fjacs])
                for i in range(njac))

            return res, jacs

    def __call__(
            self,
            variable_dict: Dict[Variable, np.ndarray],
            axis: Axis = False, 
    ) -> FactorValue:
        values = self.resolve_variable_dict(variable_dict)
        val = self._call_factor(values, variables=None)
        return FactorValue(val, {})

    def func_jacobian(
            self,
            variable_dict: Dict[Variable, np.ndarray],
            variables: Optional[Tuple[Variable, ...]] = None,
            axis: Axis = False, 
            **kwargs
    ) -> Tuple[FactorValue, JacobianValue]:
        """
        Call the underlying factor

        Parameters
        ----------
        args
            Positional arguments for the factor
        kwargs
            Keyword arguments for the factor

        Returns
        -------
        Object encapsulating the result of the function call
        """
        if variables is None:
            variables = self.variables

        variable_names = tuple(
            self._variable_name_kw[v.name]
            for v in variables)
        kwargs = self.resolve_variable_dict(variable_dict)
        val, jacs = self._call_factor(
            kwargs, variables=variable_names)
        val = aggregate(
            val.reshape(self._function_shape(**kwargs)), axis)
        jacobian = {
            v: aggregate(jac, axis) 
            for v, jac in zip(self._variables, jacs)
        }
        return FactorValue(val, {}), JacobianValue(jacobian, {})

    def __eq__(self, other: Union["Factor", Variable]):
        """
        If set equal to a variable that variable is taken to be deterministic and
        so a DeterministicFactorNode is generated.
        """
        if isinstance(other, Factor):
            if isinstance(other, type(self)):
                return (
                        (self._factor == other._factor)
                        and (frozenset(self._kwargs.items())
                             == frozenset(other._kwargs.items()))
                        and (frozenset(self.variables)
                             == frozenset(other.variables))
                        and (frozenset(self.deterministic_variables)
                             == frozenset(self.deterministic_variables)))
            else:
                return False

        from autofit.graphical.factor_graphs.graph import \
            DeterministicFactorJacobianNode
        return DeterministicFactorJacobianNode(
            self._factor_jacobian,
            other,
            variable_order=self._variables,
            **self._kwargs
        )

    def __repr__(self) -> str:
        args = ", ".join(chain(
            map("{0[0]}={0[1]}".format, self._kwargs.items())))
        return f"FactorJacobian({self.name}, {args})"