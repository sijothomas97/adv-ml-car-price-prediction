"""Row-level cleaning / feature engineering, consolidated from the notebooks.

These steps mirror ``02_nb_data_preprocessing`` and ``03_nb_feature_engineering``:

1. Drop noisy rows (missing ``reg_code`` on non-NEW vehicles).
2. Impute ``reg_code`` for NEW vehicles and derive missing ``year_of_registration``
   from the numeric registration code.
3. Impute mileage (per-year mean), colour / body / fuel type (mode).
4. Remove mileage outliers with the IQR rule.
5. Engineer ``age`` from ``year_of_registration`` and drop identifier columns.

All functions are pure (copy in, copy out) so they are easy to test.
"""

from datetime import date

import numpy as np
import pandas as pd

from car_price import config as c

# Two-digit UK plate codes: "51".."70" -> Sep 2001..2020, "05" -> 2005, etc.
_PLATE_CODE_PIVOT = 50


def remove_noise(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows with a missing reg_code on vehicles that are not NEW."""
    mask = df[c.COL_REG_CODE].isna() & (df[c.COL_VHL_COND] != c.VAL_NEW)
    return df.loc[~mask].copy()


def impute_reg_code(df: pd.DataFrame) -> pd.DataFrame:
    """NEW vehicles have no reg code yet; assign the current-period plate code."""
    df = df.copy()
    mask = df[c.COL_REG_CODE].isna() & (df[c.COL_VHL_COND] == c.VAL_NEW)
    df.loc[mask, c.COL_REG_CODE] = "71"
    return df


def impute_reg_year(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing ``year_of_registration`` from the numeric plate code.

    Codes below 50 map to 2000 + code, codes >= 50 map to 2000 + code - 50.
    Non-numeric (letter) codes are left as NaN and imputed with the median year.
    """
    df = df.copy()
    code_num = pd.to_numeric(df[c.COL_REG_CODE], errors="coerce").astype("float64")
    derived = np.where(
        code_num < _PLATE_CODE_PIVOT, 2000 + code_num, 2000 + code_num - _PLATE_CODE_PIVOT
    )
    year = df[c.COL_REG_YEAR].fillna(pd.Series(derived, index=df.index))
    year = year.fillna(year.median())
    df[c.COL_REG_YEAR] = year.astype(float)
    return df


def impute_mileage(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing mileage with the mean mileage of same-year vehicles."""
    df = df.copy()
    per_year_mean = df.groupby(c.COL_REG_YEAR)[c.COL_MILEAGE].transform("mean")
    df[c.COL_MILEAGE] = df[c.COL_MILEAGE].fillna(per_year_mean)
    df[c.COL_MILEAGE] = df[c.COL_MILEAGE].fillna(df[c.COL_MILEAGE].median())
    return df


def impute_modes(df: pd.DataFrame) -> pd.DataFrame:
    """Fill missing colour / body type / fuel type with the column mode."""
    df = df.copy()
    for col in (c.COL_STD_COLR, c.COL_BD_TYPE, c.COL_FL_TYPE):
        if df[col].isna().any():
            df[col] = df[col].fillna(df[col].mode()[0])
    return df


def remove_mileage_outliers(df: pd.DataFrame) -> pd.DataFrame:
    """Drop rows outside 1.5 * IQR of mileage."""
    q1 = df[c.COL_MILEAGE].quantile(0.25)
    q3 = df[c.COL_MILEAGE].quantile(0.75)
    iqr = q3 - q1
    lower, upper = q1 - 1.5 * iqr, q3 + 1.5 * iqr
    mask = df[c.COL_MILEAGE].between(lower, upper)
    return df.loc[mask].copy()


def add_age(df: pd.DataFrame, current_year: int | None = None) -> pd.DataFrame:
    """Add ``age`` (years since registration) and drop ``year_of_registration``."""
    df = df.copy()
    year = current_year if current_year is not None else date.today().year
    df[c.COL_AGE] = (year - df[c.COL_REG_YEAR]).clip(lower=0)
    return df.drop(columns=[c.COL_REG_YEAR])


def clean(df: pd.DataFrame, current_year: int | None = None) -> pd.DataFrame:
    """Full cleaning pass: returns a model-ready frame of FEATURES + TARGET."""
    df = remove_noise(df)
    df = impute_reg_code(df)
    df = impute_reg_year(df)
    df = impute_mileage(df)
    df = impute_modes(df)
    df = remove_mileage_outliers(df)
    df = add_age(df, current_year=current_year)
    df[c.COL_VHL_TYPE] = (
        df[c.COL_VHL_TYPE].astype(str).str.lower().eq("true").astype(int)
    )
    keep = [col for col in c.FEATURES + [c.TARGET] if col in df.columns]
    return df[keep].reset_index(drop=True)
