"""
================================================================================
INSTRUMENTAL VARIABLE ANALYSIS (2SLS) — PUBLICATION READY VERSION
Causal Identification: Religion -> CO2 Emissions

Features:
- IV2SLS (linearmodels)
- First-stage F-statistic
- Durbin-Wu-Hausman test
- Hansen J test (overidentification)
- Robust SE
- LaTeX export
- Safe key handling (no KeyError)
================================================================================
"""

import os
import numpy as np
import pandas as pd
import matplotlib.pyplot as plt

from linearmodels.iv import IV2SLS
from statsmodels.api import OLS, add_constant


# ==============================
# 1. Load Data
# ==============================
DATA_PATH = "data/processed/merged_2020.csv"

df = pd.read_csv(DATA_PATH)
print(f"[INFO] Data shape: {df.shape}")


# ==============================
# 2. Instruments (Geography)
# ==============================
latitude_dict = {
    "Germany": 51.17, "France": 46.60, "United States": 37.77,
    "Saudi Arabia": 23.89, "India": 20.59, "China": 35.86,
    "Japan": 36.20, "Brazil": -14.24, "Nigeria": 9.08,
    "Egypt": 26.82, "Turkey": 39.03
}

longitude_dict = {
    "Germany": 10.45, "France": 2.21, "United States": -100.0,
    "Saudi Arabia": 45.08, "India": 78.96, "China": 104.20,
    "Japan": 138.25, "Brazil": -51.93, "Nigeria": 8.68,
    "Egypt": 30.80, "Turkey": 35.24
}

MECCA_LAT, MECCA_LON = 21.3891, 39.8579


def haversine(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371.0
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1 - a))


df["latitude"] = df["Country"].map(latitude_dict)
df["longitude"] = df["Country"].map(longitude_dict)

df["distance_to_mecca"] = df.apply(
    lambda r: haversine(r["latitude"], r["longitude"], MECCA_LAT, MECCA_LON) / 1000
    if pd.notnull(r["latitude"]) and pd.notnull(r["longitude"])
    else np.nan,
    axis=1
)


# ==============================
# 3. Clean Data
# ==============================
cols = ["Muslims", "gdp", "co2_per_capita", "latitude", "distance_to_mecca"]
df = df.dropna(subset=cols).copy()

df["log_gdp"] = np.log(df["gdp"])
df["log_co2"] = np.log(df["co2_per_capita"] + 1e-8)

print(f"[INFO] Clean sample size: {len(df)}")


# ==============================
# 4. Define Variables
# ==============================
y = df["log_co2"]
endog = df["Muslims"]
exog = add_constant(df[["log_gdp"]])
instr = df[["latitude", "distance_to_mecca"]]


# ==============================
# 5. 2SLS Estimation
# ==============================
iv_model = IV2SLS(dependent=y, exog=exog, endog=endog, instruments=instr)
iv_res = iv_model.fit(cov_type="robust")


# ==============================
# 6. OLS Benchmark
# ==============================
ols_X = add_constant(df[["Muslims", "log_gdp"]])
ols_res = OLS(y, ols_X).fit(cov_type="HC3")


# ==============================
# 7. First Stage (manual)
# ==============================
fs_X = add_constant(pd.concat([instr, df[["log_gdp"]]], axis=1))
fs_res = OLS(endog, fs_X).fit()
f_stat = fs_res.fvalue


# ==============================
# 8. Diagnostics (SAFE ACCESS)
# ==============================

# Durbin-Wu-Hausman
dwh = iv_res.wu_hausman()

# Hansen J (overidentification)
hansen = iv_res.sargan   # (linearmodels gives Sargan/Hansen form)


def safe(params, key):
    return params.get(key, np.nan)


# ==============================
# 9. Print Results
# ==============================
print("\n================ RESULTS ================")
print(f"First-stage F: {f_stat:.3f}")
print(f"DWH p-value: {dwh.pvalue:.4f}")
print(f"Hansen J p-value: {hansen.pval:.4f}")

print("\n--- OLS ---")
print(ols_res.summary().tables[1])

print("\n--- IV ---")
print(iv_res.summary)


# ==============================
# 10. LaTeX Export
# ==============================
os.makedirs("outputs/tables", exist_ok=True)

tex = f"""
\\begin{{table}}[htbp]
\\centering
\\caption{{IV Estimation Results}}
\\begin{{tabular}}{{lcc}}
\\hline
 & OLS & 2SLS \\\\
\\hline
Muslims
& {safe(ols_res.params, "Muslims"):.4f}
& {safe(iv_res.params, "Muslims"):.4f} \\\\
log GDP
& {safe(ols_res.params, "log_gdp"):.4f}
& {safe(iv_res.params, "log_gdp"):.4f} \\\\
\\hline
Observations & {int(ols_res.nobs)} & {int(iv_res.nobs)} \\\\
First-stage F & -- & {f_stat:.2f} \\\\
DWH p-value & -- & {dwh.pvalue:.4f} \\\\
Hansen p-value & -- & {hansen.pval:.4f} \\\\
\\hline
\\end{{tabular}}
\\end{{table}}
"""

with open("outputs/tables/iv.tex", "w", encoding="utf-8") as f:
    f.write(tex)


# ==============================
# 11. Plot
# ==============================
os.makedirs("outputs/figures", exist_ok=True)

plt.scatter(df["Muslims"], df["log_co2"], alpha=0.5)
plt.xlabel("Muslims")
plt.ylabel("log CO2")
plt.title("Religion vs CO2")
plt.grid()
plt.savefig("outputs/figures/scatter.png", dpi=300)
plt.close()


print("\n[OK] Pipeline finished successfully")