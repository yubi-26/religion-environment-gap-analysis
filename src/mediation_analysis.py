"""
======================================================================
MEDIATION ANALYSIS
Religion -> Institutions -> Environmental Performance

Pathway:
Muslim population share
        |
        v
Rule of Law / Political Rights
        |
        v
Environmental Performance (EPI)

======================================================================
"""


import os
import pandas as pd
import numpy as np
import statsmodels.api as sm


print("="*70)
print("MEDIATION ANALYSIS START")
print("="*70)


# =====================================================
# Load data
# =====================================================

df = pd.read_csv(
    "data/processed/merged_epi_full_2020.csv"
)


print("Original shape:", df.shape)



# =====================================================
# Prepare variables
# =====================================================

required = [
    "Muslims",
    "gdp",
    "epi_score",
    "rule_of_law",
    "political_rights",
    "social_progress"
]


df = df[required].copy()


df["log_gdp"] = np.log(
    df["gdp"]
)


print("\nMissing values:")
print(df.isna().sum())



# =====================================================
# Helper function
# =====================================================

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



# =====================================================
# PATH 1
# Religion -> Institution
# =====================================================


print("\n")
print("="*70)
print("MODEL 1")
print("Muslims -> Rule of Law")
print("="*70)


df1 = df.dropna(
    subset=[
        "Muslims",
        "rule_of_law",
        "log_gdp"
    ]
)


model1 = regression(
    df1["rule_of_law"],
    df1[
        [
            "Muslims",
            "log_gdp"
        ]
    ]
)



# =====================================================
# PATH 2
# Institution -> EPI
# =====================================================


print("\n")
print("="*70)
print("MODEL 2")
print("Rule of Law -> EPI")
print("="*70)


df2 = df.dropna(
    subset=[
        "epi_score",
        "rule_of_law",
        "log_gdp"
    ]
)


model2 = regression(
    df2["epi_score"],
    df2[
        [
            "rule_of_law",
            "log_gdp"
        ]
    ]
)



# =====================================================
# PATH 3
# Full mediation model
# =====================================================


print("\n")
print("="*70)
print("MODEL 3")
print("Muslims + Rule of Law -> EPI")
print("="*70)


df3 = df.dropna(
    subset=[
        "epi_score",
        "Muslims",
        "rule_of_law",
        "log_gdp"
    ]
)


model3 = regression(
    df3["epi_score"],
    df3[
        [
            "Muslims",
            "rule_of_law",
            "log_gdp"
        ]
    ]
)



# =====================================================
# Compare coefficients
# =====================================================


print("\n")
print("="*70)
print("MEDIATION SUMMARY")
print("="*70)


print(
    "Model 1 Muslims -> Rule of Law:",
    model1.params["Muslims"],
    "p=",
    model1.pvalues["Muslims"]
)


print(
    "Model 2 Rule of Law -> EPI:",
    model2.params["rule_of_law"],
    "p=",
    model2.pvalues["rule_of_law"]
)


print(
    "Model 3 Muslims -> EPI:",
    model3.params["Muslims"],
    "p=",
    model3.pvalues["Muslims"]
)



# =====================================================
# Save results
# =====================================================


os.makedirs(
    "outputs/tables",
    exist_ok=True
)


with open(
    "outputs/tables/mediation_results.txt",
    "w",
    encoding="utf-8"
) as f:


    f.write(
        "MEDIATION ANALYSIS RESULTS\n\n"
    )


    f.write(
        "MODEL 1: Muslims -> Rule of Law\n"
    )

    f.write(
        str(model1.summary())
    )


    f.write(
        "\n\nMODEL 2: Rule of Law -> EPI\n"
    )

    f.write(
        str(model2.summary())
    )


    f.write(
        "\n\nMODEL 3: Muslims + Rule of Law -> EPI\n"
    )


    f.write(
        str(model3.summary())
    )



print("\n")
print("="*70)
print("DONE")
print("="*70)

print(
    "Saved:"
)

print(
    "outputs/tables/mediation_results.txt"
)