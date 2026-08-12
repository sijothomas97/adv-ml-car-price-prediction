import numpy as np
import pandas as pd
import pytest

from car_price import config as c


@pytest.fixture
def raw_df() -> pd.DataFrame:
    """Small synthetic frame mimicking the raw adverts CSV (with NaNs/noise)."""
    rng = np.random.default_rng(0)
    n = 120
    df = pd.DataFrame(
        {
            c.COL_PUBLIC_REF: np.arange(n),
            c.COL_MILEAGE: rng.uniform(1_000, 90_000, n).round(0),
            c.COL_REG_CODE: pd.array(
                [str(v) for v in rng.integers(2, 71, n)], dtype="string"
            ),
            c.COL_STD_COLR: rng.choice(["Blue", "Red", "Grey", "Black"], n),
            c.COL_STD_MK: rng.choice(["Ford", "BMW", "Audi"], n),
            c.COL_STD_MDL: rng.choice(["Fiesta", "3 Series", "A4"], n),
            c.COL_VHL_COND: ["USED"] * n,
            c.COL_REG_YEAR: rng.integers(2002, 2021, n).astype(float),
            c.COL_PRICE: rng.uniform(2_000, 40_000, n).round(0),
            c.COL_BD_TYPE: rng.choice(["Hatchback", "SUV", "Saloon"], n),
            c.COL_VHL_TYPE: rng.choice([True, False], n),
            c.COL_FL_TYPE: rng.choice(["Petrol", "Diesel"], n),
        }
    )
    # Inject the messiness the cleaning steps must handle
    df.loc[0, c.COL_REG_CODE] = pd.NA  # noise: USED with no reg code -> dropped
    df.loc[1, [c.COL_VHL_COND, c.COL_REG_CODE, c.COL_REG_YEAR, c.COL_MILEAGE]] = [
        "NEW",
        pd.NA,
        np.nan,
        0.0,
    ]
    df.loc[2, c.COL_REG_YEAR] = np.nan  # derived from reg_code
    df.loc[3, c.COL_MILEAGE] = np.nan  # per-year mean impute
    df.loc[4, c.COL_STD_COLR] = np.nan  # mode impute
    df.loc[5, c.COL_BD_TYPE] = np.nan
    df.loc[6, c.COL_FL_TYPE] = np.nan
    df.loc[7, c.COL_MILEAGE] = 900_000  # outlier -> dropped
    return df
