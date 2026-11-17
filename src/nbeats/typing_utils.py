from typing import Any, TypeAlias

import numpy as np
from numpy.typing import NDArray

RealArray: TypeAlias = NDArray[np.floating[Any] | np.integer[Any]]
FloatArray: TypeAlias = NDArray[np.floating[Any]]
