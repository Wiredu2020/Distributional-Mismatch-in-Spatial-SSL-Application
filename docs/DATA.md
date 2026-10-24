# Data preparation

No external dataset is required for synthetic experiments. The synthetic
configuration records grid size, sample sizes, seeds, and sampling parameters.
The clean repository does not bundle real-data caches.

| Dataset | Local prerequisites and retrieval |
| --- | --- |
| Housing | `load_housing` calls scikit-learn's California Housing fetcher; county centroids are fetched and cached under `data/real_world/housing/us_county_centroids.csv`. Internet is required on first use. |
| Socio-economic | Place `unemp_income.csv`, `poverty.xlsx`, `population.xlsx`, and `edu.xlsx` under `data/real_world/socioeconomic/`. County centroids are cached there as `us_county_centroids.csv`. These four tables are read from disk; the loader does not download them. |
| Air quality | Place `annual_conc_by_monitor_2023.csv` under `data/real_world/air_quality/`; the loader also needs `data/real_world/socioeconomic/population.xlsx`. |
| PovertyMap-WILDS | The runner uses `data/wilds_povertymap/`, metadata and imagery retrieval functions, and a feature cache. The CodaLab endpoints in `src/ssl_spatial/data/wilds_poverty.py` must be reachable, or the corresponding documented cached files must already exist. |

Sources:

- USDA county tables: https://www.ers.usda.gov/data-products/county-level-data-sets
- EPA annual summaries: https://aqs.epa.gov/aqsweb/airdata/download_files.html
- WILDS: https://wilds.stanford.edu/
- California Housing: scikit-learn's `fetch_california_housing` dataset interface.

The socio-economic loader expects these column names:

- CSV: `FIPS_Code`, `State`, `Unemployment_rate_2023`, `Median_Household_Income_2022`.
- Poverty: `FIPS_Code`, `PCTPOVALL_2023`; four header rows are skipped.
- Population: `FIPStxt`, `POP_ESTIMATE_2023`; four header rows are skipped.
- Education: `FIPS Code`, `Percent of adults with a bachelor's degree or higher, 2019-23`;
  three header rows are skipped.

Thus this is not a uniformly single-year set of covariates: income is from 2022,
and education covers 2019–2023. Retain the exact input release and checksums when
archiving results. Do not substitute newer tables silently. The author-provided
working archive contains these tables, but redistribution permission and a stable
public data manifest should be confirmed before adding them to GitHub.

The EPA loader selects `PM2.5 - Local Conditions`, groups by monitoring site,
and joins ozone and PM10 summaries. It median-imputes missing pollutant covariates
and drops sites lacking county population. Review the source for the exact PM2.5
sample duration and required columns.

All real-data labels use a full-dataset median split. Housing groups use nearest
county centroids, not a polygon containment operation. These choices are part of
the benchmark definition and should be retained for comparable results.
