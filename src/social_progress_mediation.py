import pandas as pd
import numpy as np
import statsmodels.api as sm
import os


print("="*70)
print("SOCIAL PROGRESS MEDIATION ANALYSIS START")
print("="*70)


# ===============================
# Load data
# ===============================

df = pd.read_csv(
    "data/processed/merged_epi_full_2020.csv"
)


print("Original shape:", df.shape)


# ===============================
# Prepare variables
# ===============================

df["log_gdp"] = np.log(df["gdp"])


variables = [
    "Muslims",
    "social_progress",
    "epi_score",
    "log_gdp"
]


df_model = df[variables].dropna()


print("\nAnalysis sample:", len(df_model))

print("\nMissing:")
print(df[variables].isna().sum())



# ===============================
# Model 1
# Muslims -> Social Progress
# ===============================

print("\n")
print("="*70)
print("MODEL 1")
print("Muslims -> Social Progress")
print("="*70)


X1 = df_model[
    [
        "Muslims",
        "log_gdp"
    ]
]

X1 = sm.add_constant(X1)

y1 = df_model["social_progress"]


model1 = sm.OLS(
    y1,
    X1
).fit(
    cov_type="HC3"
)


print(model1.summary())



# ===============================
# Model 2
# Social Progress -> EPI
# ===============================

print("\n")
print("="*70)
print("MODEL 2")
print("Social Progress -> EPI")
print("="*70)


X2 = df_model[
    [
        "social_progress",
        "log_gdp"
    ]
]

X2 = sm.add_constant(X2)


y2 = df_model["epi_score"]


model2 = sm.OLS(
    y2,
    X2
).fit(
    cov_type="HC3"
)


print(model2.summary())



# ===============================
# Model 3
# Muslims + Social Progress -> EPI
# ===============================

print("\n")
print("="*70)
print("MODEL 3")
print("Muslims + Social Progress -> EPI")
print("="*70)


X3 = df_model[
    [
        "Muslims",
        "social_progress",
        "log_gdp"
    ]
]

X3 = sm.add_constant(X3)


y3 = df_model["epi_score"]


model3 = sm.OLS(
    y3,
    X3
).fit(
    cov_type="HC3"
)


print(model3.summary())



# ===============================
# Summary
# ===============================

print("\n")
print("="*70)
print("MEDIATION SUMMARY")
print("="*70)


print(
    "Model1 Muslims -> SPI:",
    model1.params["Muslims"],
    "p=",
    model1.pvalues["Muslims"]
)


print(
    "Model2 SPI -> EPI:",
    model2.params["social_progress"],
    "p=",
    model2.pvalues["social_progress"]
)


print(
    "Model3 Muslims -> EPI:",
    model3.params["Muslims"],
    "p=",
    model3.pvalues["Muslims"]
)


print(
    "Model3 SPI -> EPI:",
    model3.params["social_progress"],
    "p=",
    model3.pvalues["social_progress"]
)



# Save result

os.makedirs(
    "outputs/tables",
    exist_ok=True
)


with open(
    "outputs/tables/social_progress_mediation_results.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        str(model1.summary())
    )

    f.write(
        "\n\n"
    )

    f.write(
        str(model2.summary())
    )

    f.write(
        "\n\n"
    )

    f.write(
        str(model3.summary())
    )


print("\n")
print("="*70)
print("DONE")
print("="*70)

print(
    "Saved:",
    "outputs/tables/social_progress_mediation_results.txt"
)