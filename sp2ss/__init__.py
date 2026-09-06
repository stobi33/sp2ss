"""sp2ss -- Touchstone S-parameters to a discrete state-space PDN macromodel
suitable for real-number-modelling (RNM) simulation."""

__version__ = "1.0.0"

from .touchstone import Network, read_touchstone, write_touchstone      # noqa: F401
from .convert import to_param, subset_ports, dc_extrapolate             # noqa: F401
from .vectfit import vector_fit, start_poles, model_response            # noqa: F401
from .statespace import (StateSpace, DiscreteSS, from_pole_residue,     # noqa: F401
                         balanced_truncate, to_modal, discretize)
from .passivity import check_passivity, enforce_passivity               # noqa: F401
