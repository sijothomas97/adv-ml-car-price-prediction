"""Loading the raw adverts dataset."""

from pathlib import Path

import pandas as pd

from car_price import config


def load_raw(path: str | Path = config.DATA_CSV) -> pd.DataFrame:
    """Read the raw adverts CSV.

    ``reg_code`` is read as string: UK registration codes can be letters
    (pre-2001 plates) or numbers, and mixing types breaks downstream logic.
    """
    return pd.read_csv(path, dtype={config.COL_REG_CODE: "string"})
