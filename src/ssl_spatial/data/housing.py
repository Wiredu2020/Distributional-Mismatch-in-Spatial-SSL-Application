"""Housing / built-environment real-world dataset (manuscript methods.tex
Section 2.2.3, first bullet): California Housing
\\citep{pace1997sparse} -- 1990 census block-group median house values,
paired with block-group covariates (income, age, rooms, occupancy) and
centroid coordinates. Loaded via `sklearn.datasets.fetch_california_housing`
(auto-downloaded and cached by scikit-learn; no manual step required).

Block groups are assigned to their nearest California county (by centroid
distance, using the free county-centroid lookup in
`ssl_spatial.data.region_mismatch`) so that county acts as the "region" for
the mismatch-severity sweep, mirroring how WILDS PovertyMap uses country.
"""
from __future__ import annotations

import numpy as np
import pandas as pd

from ssl_spatial.data.region_mismatch import assign_nearest_region, download_us_county_centroids

FEATURE_NAMES = ["MedInc", "HouseAge", "AveRooms", "AveBedrms", "Population", "AveOccup"]


def load_housing(data_dir) -> pd.DataFrame:
    """Returns a DataFrame with FEATURE_NAMES, `label` (binary, above/below
    median house value), `lat`, `lon`, and `region` (nearest CA county).
    """
    from sklearn.datasets import fetch_california_housing

    data = fetch_california_housing(as_frame=True)
    df = data.frame.reset_index(drop=True)
    df["label"] = (df["MedHouseVal"] > df["MedHouseVal"].median()).astype(int)
    df["lat"] = df["Latitude"]
    df["lon"] = df["Longitude"]

    centroids = download_us_county_centroids(data_dir / "us_county_centroids.csv")
    ca_centroids = centroids[centroids["fips_code"].str.startswith("06")].copy()
    coords = df[["lat", "lon"]].to_numpy()
    df["region"] = assign_nearest_region(coords, ca_centroids, lat_col="lat", lon_col="lng", label_col="name")

    return df[["lat", "lon", "region", "label"] + FEATURE_NAMES]


__all__ = ["load_housing", "FEATURE_NAMES"]
