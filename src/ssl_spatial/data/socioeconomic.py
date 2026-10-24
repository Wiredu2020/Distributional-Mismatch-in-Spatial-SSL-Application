"""Urban socio-economic real-world dataset (manuscript methods.tex Section
2.2.3, second bullet): US county-level poverty, income, unemployment,
education, and population, from the USDA Economic Research Service's
County-level Data Sets (derived from Census/BLS source data; direct CSV/XLSX
download, no API key or manual step required) \\citep{usda2024county}, joined
to county centroid coordinates.

State is used as the "region" for the mismatch-severity sweep -- a real,
administrative grouping directly analogous to WILDS PovertyMap's country
grouping.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

from ssl_spatial.data.region_mismatch import download_us_county_centroids

FEATURE_NAMES = ["unemployment_rate", "log_median_income", "log_population", "pct_bachelors"]


def _read_unemployment_income(path: Path) -> pd.DataFrame:
    df = pd.read_csv(path, dtype={"FIPS_Code": str})
    df["FIPS_Code"] = df["FIPS_Code"].str.zfill(5)
    wide = df.pivot_table(index=["FIPS_Code", "State"], columns="Attribute", values="Value", aggfunc="first")
    return wide[["Unemployment_rate_2023", "Median_Household_Income_2022"]].reset_index()


def _read_poverty(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, skiprows=4, dtype={"FIPS_Code": str})
    df["FIPS_Code"] = df["FIPS_Code"].astype(str).str.zfill(5)
    return df[["FIPS_Code", "PCTPOVALL_2023"]]


def _read_population(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, skiprows=4, dtype={"FIPStxt": str})
    df["FIPS_Code"] = df["FIPStxt"].astype(str).str.zfill(5)
    return df[["FIPS_Code", "POP_ESTIMATE_2023"]]


def _read_education(path: Path) -> pd.DataFrame:
    df = pd.read_excel(path, skiprows=3, dtype={"FIPS Code": str})
    df["FIPS_Code"] = df["FIPS Code"].astype(str).str.zfill(5)
    col = "Percent of adults with a bachelor's degree or higher, 2019-23"
    return df[["FIPS_Code", col]].rename(columns={col: "pct_bachelors"})


def load_socioeconomic(data_dir) -> pd.DataFrame:
    """Returns a DataFrame with FEATURE_NAMES, `label` (binary, above/below
    median poverty rate), `lat`, `lon`, and `region` (US state).
    """
    data_dir = Path(data_dir)
    unemp = _read_unemployment_income(data_dir / "unemp_income.csv")
    poverty = _read_poverty(data_dir / "poverty.xlsx")
    population = _read_population(data_dir / "population.xlsx")
    education = _read_education(data_dir / "edu.xlsx")

    df = unemp.merge(poverty, on="FIPS_Code").merge(population, on="FIPS_Code").merge(education, on="FIPS_Code")

    # Drop national ("00000") and state-total rows (FIPS ending "000"), and any
    # county with missing values in the fields used here.
    df = df[(df["FIPS_Code"] != "00000") & (~df["FIPS_Code"].str.endswith("000"))]
    df = df.dropna(subset=["Unemployment_rate_2023", "Median_Household_Income_2022",
                            "PCTPOVALL_2023", "POP_ESTIMATE_2023", "pct_bachelors"])

    centroids = download_us_county_centroids(data_dir / "us_county_centroids.csv")
    df = df.merge(centroids[["fips_code", "lat", "lng"]], left_on="FIPS_Code", right_on="fips_code")

    df["label"] = (df["PCTPOVALL_2023"] > df["PCTPOVALL_2023"].median()).astype(int)
    df["unemployment_rate"] = df["Unemployment_rate_2023"].astype(float)
    df["log_median_income"] = np.log(df["Median_Household_Income_2022"].astype(float))
    df["log_population"] = np.log(df["POP_ESTIMATE_2023"].astype(float))
    df["pct_bachelors"] = df["pct_bachelors"].astype(float)
    df["region"] = df["State"]
    df["lon"] = df["lng"]

    df = df.reset_index(drop=True)
    return df[["lat", "lon", "region", "label"] + FEATURE_NAMES]


__all__ = ["load_socioeconomic", "FEATURE_NAMES"]
