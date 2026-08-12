import numpy as np
import pandas as pd

from car_price import config as c
from car_price import preprocessing as pp


def test_remove_noise_drops_used_rows_without_reg_code(raw_df):
    out = pp.remove_noise(raw_df)
    assert 0 not in out.index  # USED + missing reg_code dropped
    assert 1 in out.index  # NEW + missing reg_code kept


def test_impute_reg_code_fills_new_vehicles(raw_df):
    out = pp.impute_reg_code(pp.remove_noise(raw_df))
    assert out.loc[1, c.COL_REG_CODE] == "71"
    assert not out[c.COL_REG_CODE].isna().any()


def test_impute_reg_year_derives_from_plate_code():
    df = pd.DataFrame(
        {
            c.COL_REG_CODE: pd.array(["18", "68", "05", "Y"], dtype="string"),
            c.COL_REG_YEAR: [np.nan, np.nan, 2005.0, np.nan],
        }
    )
    out = pp.impute_reg_year(df)
    assert out.loc[0, c.COL_REG_YEAR] == 2018.0  # code < 50 -> 2000 + code
    assert out.loc[1, c.COL_REG_YEAR] == 2018.0  # code >= 50 -> 2000 + code - 50
    assert out.loc[2, c.COL_REG_YEAR] == 2005.0  # existing value untouched
    assert not out[c.COL_REG_YEAR].isna().any()  # letter code -> median fallback


def test_impute_mileage_uses_per_year_mean():
    df = pd.DataFrame(
        {
            c.COL_REG_YEAR: [2015.0, 2015.0, 2015.0],
            c.COL_MILEAGE: [10_000.0, 20_000.0, np.nan],
        }
    )
    out = pp.impute_mileage(df)
    assert out.loc[2, c.COL_MILEAGE] == 15_000.0


def test_remove_mileage_outliers(raw_df):
    out = pp.remove_mileage_outliers(raw_df.dropna(subset=[c.COL_MILEAGE]))
    assert 7 not in out.index
    assert len(out) < len(raw_df)


def test_add_age_replaces_year():
    df = pd.DataFrame({c.COL_REG_YEAR: [2020.0, 2024.0]})
    out = pp.add_age(df, current_year=2024)
    assert list(out[c.COL_AGE]) == [4.0, 0.0]
    assert c.COL_REG_YEAR not in out.columns


def test_clean_end_to_end(raw_df):
    out = pp.clean(raw_df, current_year=2026)
    # model-ready: only features + target, no NaNs anywhere
    assert list(out.columns) == c.FEATURES + [c.TARGET]
    assert not out.isna().any().any()
    # crossover flag encoded to 0/1 ints
    assert set(out[c.COL_VHL_TYPE].unique()) <= {0, 1}
    # noise and outlier rows removed
    assert len(out) <= len(raw_df) - 2
