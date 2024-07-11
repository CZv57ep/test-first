from datetime import datetime
from typing import Any

import numpy as np
from icecream import argumentToString, ic, install  # type: ignore
from numpy.typing import NDArray

np.set_printoptions(precision=18)


def timestamper() -> str:
    return f"{datetime.now().strftime("%F %T.%f")} |>  "


@argumentToString.register(np.ndarray)  # type: ignore
def _(_obj: NDArray[Any]) -> str:
    return f"ndarray, shape={_obj.shape}, dtype={_obj.dtype}"


ic.configureOutput(prefix=timestamper, includeContext=True)
install()
