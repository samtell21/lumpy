from .constructors import *
from .constructors import I  # explicit: la.I works, but not in __all__
from .core import *
from .spaces import *
from .geometry import *
from .decompositions import *
from .utils import *
from .equations import *

from numpy.linalg import qr, eig, inv, pinv, det

from .constructors import __all__ as _constructors_all
from .core import __all__ as _core_all
from .spaces import __all__ as _spaces_all
from .geometry import __all__ as _geometry_all
from .decompositions import __all__ as _decompositions_all
from .utils import __all__ as _utils_all
from .equations import __all__ as _equations_all

__all__ = [
    *_constructors_all,
    *_core_all,
    *_spaces_all,
    *_geometry_all,
    *_decompositions_all,
    *_utils_all,
    *_equations_all,
    "qr", "eig", "inv", "pinv", "det",
]
