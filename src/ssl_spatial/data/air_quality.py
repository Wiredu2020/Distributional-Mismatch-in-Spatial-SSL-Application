"""Environmental / air-quality monitoring real-world dataset (manuscript
methods.tex Section 2.2.3, third bullet): EPA Air Quality System (AQS)
annual monitor summary data \\citep{epa2024aqs} -- regulatory PM2.5 monitors,
which are non-randomly sited and concentrated in urban/industrial areas,
paired with co-located ozone and PM10 concentrations and county population
(from the socio-economic dataset's USDA source) as land-use-regression-style
covariates. Downloaded directly as a pre-generated CSV (no API key
required; the live AQS Data Mart API does require a key, so the static
annual-summary file is used instead).

State is used as the "region" for the mismatch-severity sweep, as in
`ssl_spatial.data.socioeconomic`.
"""
from __future__ import annotations

from pathlib import Path

import numpy as np
import pandas as pd

FEATURE_NAMES = ["ozone_mean", "pm10_mean", "log_county_population"]

_PM25_DURATION = "24-HR BLK AVG"


def _site_id(df: pd.DataFrame) -> pd.Series:
    return (df["State Code"].astype(str) + "-" + df["County Code"].astype(str).str.zfill(3)
            + "-" + df["Site Num"].astype(str).str.zfill(4))


def load_air_quality(data_dir, socioeconomic_data_dir) -> pd.DataFrame:
    """Returns a DataFrame with FEATURE_NAMES, `label` (binary, above/below
    median annual PM2.5), `lat`, `lon`, and `region` (US state).
    """
    data_dir = Path(data_dir)
    df = pd.read_csv(data_dir / "annual_conc_by_monitor_2023.csv")
    df["site_id"] = _site_id(df)

    pm25 = df[(df["Parameter Name"] == "PM2.5 - Local Conditions") & (df["Sample Duration"] == _PM25_DURATION)]
    sites = pm25.groupby("site_id").agg(
        lat=("Latitude", "first"), lon=("Longitude", "first"),
        region=("State Name", "first"), county=("County Name", "first"),
        state_code=("State Code", "first"), county_code=("County Code", "first"),
        pm25_mean=("Arithmetic Mean", "mean"),
    ).reset_index()

    for param, col in [("Ozone", "ozone_mean"), ("PM10 Total 0-10um STP", "pm10_mean")]:
        cov = df[df["Parameter Name"] == param].groupby("site_id")["Arithmetic Mean"].mean()
        sites[col] = sites["site_id"].map(cov)
    sites["ozone_mean"] = sites["ozone_mean"].fillna(sites["ozone_mean"].median())
    sites["pm10_mean"] = sites["pm10_mean"].fillna(sites["pm10_mean"].median())

    fips = (sites["state_code"].astype(str).str.zfill(2) + sites["county_code"].astype(str).str.zfill(3))
    population = pd.read_excel(Path(socioeconomic_data_dir) / "population.xlsx", skiprows=4,
                                dtype={"FIPStxt": str})
    population["FIPS_Code"] = population["FIPStxt"].astype(str).str.zfill(5)
    pop_map = population.set_index("FIPS_Code")["POP_ESTIMATE_2023"]
    sites["county_population"] = fips.map(pop_map)
    sites = sites.dropna(subset=["county_population"])
    sites["log_county_population"] = np.log(sites["county_population"].astype(float))

    sites["label"] = (sites["pm25_mean"] > sites["pm25_mean"].median()).astype(int)
    sites = sites[sites["region"].notna()].reset_index(drop=True)

    return sites[["lat", "lon", "region", "label"] + FEATURE_NAMES]


__all__ = ["load_air_quality", "FEATURE_NAMES"]
