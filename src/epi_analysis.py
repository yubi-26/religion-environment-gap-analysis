"""
==============================================================
EPI ANALYSIS
Religion, Economic Development and Environmental Performance

Models:
1. Muslims -> EPI
2. Muslims + GDP -> EPI
3. Region Fixed Effects

==============================================================
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import os


# =============================================================
# Load Data
# =============================================================

DATA_PATH = "data/processed/merged_epi_2020.csv"

df = pd.read_csv(DATA_PATH)

print("="*70)
print("EPI ANALYSIS START")
print("="*70)

print("Original shape:", df.shape)


# =============================================================
# Select variables
# =============================================================

cols = [
    "Country",
    "Region",
    "Muslims",
    "gdp",
    "epi_score"
]


analysis = df[cols].dropna().copy()

print("Analysis sample:", len(analysis))


# =============================================================
# Transform variables
# =============================================================

analysis["log_gdp"] = np.log(analysis["gdp"])


# =============================================================
# Model 1
# Muslims -> EPI
# =============================================================

print("\n")
print("="*70)
print("MODEL A: Muslims -> EPI")
print("="*70)


X1 = sm.add_constant(
    analysis[["Muslims"]]
)

y = analysis["epi_score"]


model1 = sm.OLS(
    y,
    X1
).fit(
    cov_type="HC3"
)


print(model1.summary())


# =============================================================
# Model 2
# Muslims + GDP
# =============================================================

print("\n")
print("="*70)
print("MODEL B: Muslims + GDP")
print("="*70)


X2 = sm.add_constant(
    analysis[
        [
            "Muslims",
            "log_gdp"
        ]
    ]
)


model2 = sm.OLS(
    y,
    X2
).fit(
    cov_type="HC3"
)


print(model2.summary())


# =============================================================
# Model 3
# Region Fixed Effects
# =============================================================

print("\n")
print("="*70)
print("MODEL C: REGION FIXED EFFECTS")
print("="*70)


region_dummies = pd.get_dummies(
    analysis["Region"],
    drop_first=True,
    dtype=float
)


X3 = pd.concat(
    [
        analysis[
            [
                "Muslims",
                "log_gdp"
            ]
        ],
        region_dummies
    ],
    axis=1
)


X3 = sm.add_constant(X3)


model3 = sm.OLS(
    y,
    X3
).fit(
    cov_type="HC3"
)


print(model3.summary())


# =============================================================
# Save results
# =============================================================

os.makedirs(
    "outputs/tables",
    exist_ok=True
)


with open(
    "outputs/tables/epi_results.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write("="*70)
    f.write("\nMODEL A\n")
    f.write(str(model1.summary()))

    f.write("\n\nMODEL B\n")
    f.write(str(model2.summary()))

    f.write("\n\nMODEL C REGION FE\n")
    f.write(str(model3.summary()))


# =============================================================
# Visualization
# =============================================================


os.makedirs(
    "outputs/figures",
    exist_ok=True
)


plt.figure(figsize=(8,6))


plt.scatter(
    analysis["Muslims"],
    analysis["epi_score"],
    alpha=0.6
)


plt.xlabel(
    "Muslim Population (%)"
)

plt.ylabel(
    "Environmental Performance Index"
)


plt.title(
    "Religion and Environmental Performance"
)


plt.grid()


plt.savefig(
    "outputs/figures/epi_regression.png",
    dpi=300,
    bbox_inches="tight"
)


plt.close()



print("\n")
print("="*70)
print("DONE")
print("="*70)

print(
    "Saved:"
)

print(
    "outputs/tables/epi_results.txt"
)

print(
    "outputs/figures/epi_regression.png"
)