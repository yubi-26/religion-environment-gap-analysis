"""
======================================================================
POLITICAL RIGHTS MEDIATION ANALYSIS

Research Question:
Does political institution mediate the relationship between
religious composition and environmental performance?

Path:
Religion (Muslims)
        |
        v
Political Rights (PRS)
        |
        v
Environmental Performance (EPI)

Models:
1. Muslims + GDP -> Political Rights
2. Political Rights + GDP -> EPI
3. Muslims + Political Rights + GDP -> EPI

======================================================================
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import os


print("=" * 70)
print("POLITICAL RIGHTS MEDIATION ANALYSIS START")
print("=" * 70)


# ============================================================
# 1. Load data
# ============================================================

DATA_PATH = "data/processed/merged_epi_full_2020.csv"

df = pd.read_csv(DATA_PATH)

print(f"Original shape: {df.shape}")


# ============================================================
# 2. Rename variables
# ============================================================

# PRS.new was renamed during merge
# Check possible names safely

possible_prs = [
    "political_rights",
    "PRS.new",
    "PRS"
]


prs_col = None

for c in possible_prs:
    if c in df.columns:
        prs_col = c
        break


if prs_col is None:
    raise ValueError(
        "Political Rights column not found. Check merged dataset."
    )


df["political_rights"] = df[prs_col]


# ============================================================
# 3. Prepare variables
# ============================================================

df["log_gdp"] = np.log(df["gdp"])

analysis_df = df[
    [
        "Muslims",
        "political_rights",
        "epi_score",
        "log_gdp"
    ]
].dropna()


print()
print(f"Analysis sample: {len(analysis_df)}")


print("\nMissing:")
print(
    df[
        [
            "Muslims",
            "political_rights",
            "epi_score",
            "log_gdp"
        ]
    ].isnull().sum()
)


# ============================================================
# Helper
# ============================================================

def regression(y, X):

    X = sm.add_constant(X)

    model = sm.OLS(
        y,
        X
    ).fit(
        cov_type="HC3"
    )

    print(model.summary())

    return model



# ============================================================
# MODEL 1
#
# Muslims -> Political Rights
# ============================================================

print("\n")
print("=" * 70)
print("MODEL 1")
print("Muslims -> Political Rights")
print("=" * 70)


model1 = regression(
    analysis_df["political_rights"],
    analysis_df[
        [
            "Muslims",
            "log_gdp"
        ]
    ]
)



# ============================================================
# MODEL 2
#
# Political Rights -> EPI
# ============================================================

print("\n")
print("=" * 70)
print("MODEL 2")
print("Political Rights -> EPI")
print("=" * 70)


model2 = regression(
    analysis_df["epi_score"],
    analysis_df[
        [
            "political_rights",
            "log_gdp"
        ]
    ]
)



# ============================================================
# MODEL 3
#
# Muslims + Political Rights -> EPI
# ============================================================

print("\n")
print("=" * 70)
print("MODEL 3")
print("Muslims + Political Rights -> EPI")
print("=" * 70)


model3 = regression(
    analysis_df["epi_score"],
    analysis_df[
        [
            "Muslims",
            "political_rights",
            "log_gdp"
        ]
    ]
)



# ============================================================
# Summary
# ============================================================

print("\n")
print("=" * 70)
print("MEDIATION SUMMARY")
print("=" * 70)


print(
    "Model 1 Muslims -> Political Rights:",
    model1.params["Muslims"],
    "p=",
    model1.pvalues["Muslims"]
)


print(
    "Model 2 Political Rights -> EPI:",
    model2.params["political_rights"],
    "p=",
    model2.pvalues["political_rights"]
)


print(
    "Model 3 Muslims -> EPI:",
    model3.params["Muslims"],
    "p=",
    model3.pvalues["Muslims"]
)


print(
    "Model 3 Political Rights -> EPI:",
    model3.params["political_rights"],
    "p=",
    model3.pvalues["political_rights"]
)



# ============================================================
# Save results
# ============================================================

os.makedirs(
    "outputs/tables",
    exist_ok=True
)


with open(
    "outputs/tables/political_mediation_results.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "POLITICAL RIGHTS MEDIATION ANALYSIS\n\n"
    )

    f.write(
        "MODEL 1\n"
    )

    f.write(
        str(model1.summary())
    )

    f.write(
        "\n\nMODEL 2\n"
    )

    f.write(
        str(model2.summary())
    )

    f.write(
        "\n\nMODEL 3\n"
    )

    f.write(
        str(model3.summary())
    )


print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    "Saved:"
)

print(
    "outputs/tables/political_mediation_results.txt"
)