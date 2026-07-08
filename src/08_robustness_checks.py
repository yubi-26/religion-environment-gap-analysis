"""
Robustness Checks
Religion and CO2 Analysis

Checks:
1. Region fixed effects
2. HC3 robust standard errors
3. Leave-one-out analysis
"""

import os
import numpy as np
import pandas as pd
import statsmodels.api as sm


# ==============================
# 1. Load data
# ==============================

INPUT = "data/processed/merged_epi_2020.csv"

df = pd.read_csv(INPUT)

print("=" * 70)
print("ROBUSTNESS CHECKS START")
print("=" * 70)

print("Original shape:", df.shape)


# ==============================
# 2. Variable preparation
# ==============================

required = [
    "Muslims",
    "co2_per_capita",
    "gdp",
    "Region"
]

missing = [
    c for c in required
    if c not in df.columns
]


if missing:
    raise ValueError(
        f"Missing columns: {missing}"
    )


analysis = df[
    required
].copy()


# remove invalid values

analysis = analysis[
    analysis["co2_per_capita"] > 0
]

analysis = analysis[
    analysis["gdp"] > 0
]


analysis = analysis.dropna()


analysis["log_co2"] = np.log(
    analysis["co2_per_capita"]
)

analysis["log_gdp"] = np.log(
    analysis["gdp"]
)


print(
    "Analysis sample:",
    len(analysis)
)


# ==============================
# 3. Model A
# Baseline
# ==============================


X_base = sm.add_constant(
    analysis[
        [
            "Muslims",
            "log_gdp"
        ]
    ]
)


y = analysis["log_co2"]


model_base = sm.OLS(
    y,
    X_base
).fit(
    cov_type="HC3"
)


print("\n")
print("=" * 70)
print("MODEL A: BASELINE HC3")
print("=" * 70)

print(model_base.summary())


# ==============================
# 4. Region Fixed Effects
# ==============================


region_dummy = pd.get_dummies(
    analysis["Region"],
    drop_first=True
)


X_fe = pd.concat(
    [
        analysis[
            [
                "Muslims",
                "log_gdp"
            ]
        ],
        region_dummy
    ],
    axis=1
)


X_fe = sm.add_constant(
    X_fe
)


# ensure numeric

X_fe = X_fe.astype(float)


model_fe = sm.OLS(
    y,
    X_fe
).fit(
    cov_type="HC3"
)


print("\n")
print("=" * 70)
print("MODEL B: REGION FIXED EFFECTS + HC3")
print("=" * 70)

print(model_fe.summary())


# ==============================
# 5. Compare coefficients
# ==============================


print("\n")
print("=" * 70)
print("COEFFICIENT COMPARISON")
print("=" * 70)


comparison = pd.DataFrame(
    {
        "Baseline_HC3":
            model_base.params,

        "Region_FE_HC3":
            model_fe.params.reindex(
                model_base.params.index
            )
    }
)


print(comparison)


# ==============================
# 6. Leave-one-out
# ==============================


oil_countries = [
    "Qatar",
    "Saudi Arabia",
    "United Arab Emirates",
    "Kuwait",
    "Brunei"
]


print("\n")
print("=" * 70)
print("LEAVE-ONE-OUT ANALYSIS")
print("=" * 70)


loo_results = []


for country in oil_countries:

    if "Country" not in df.columns:
        continue


    sub = df[
        df["Country"] != country
    ].copy()


    sub = sub[
        sub["co2_per_capita"] > 0
    ]


    sub = sub[
        sub["gdp"] > 0
    ]


    sub = sub.dropna(
        subset=[
            "Muslims",
            "co2_per_capita",
            "gdp"
        ]
    )


    sub["log_co2"] = np.log(
        sub["co2_per_capita"]
    )


    sub["log_gdp"] = np.log(
        sub["gdp"]
    )


    X = sm.add_constant(
        sub[
            [
                "Muslims",
                "log_gdp"
            ]
        ]
    )


    m = sm.OLS(
        sub["log_co2"],
        X
    ).fit(
        cov_type="HC3"
    )


    loo_results.append(
        {
            "removed_country": country,
            "muslims_coef":
                m.params.get(
                    "Muslims",
                    np.nan
                ),
            "p_value":
                m.pvalues.get(
                    "Muslims",
                    np.nan
                ),
            "sample_size":
                len(sub)
        }
    )


loo = pd.DataFrame(
    loo_results
)


print(loo)


# ==============================
# 7. Save results
# ==============================


os.makedirs(
    "outputs/tables",
    exist_ok=True
)


with open(
    "outputs/tables/robustness_results.txt",
    "w",
    encoding="utf-8"
) as f:

    f.write(
        "ROBUSTNESS CHECK RESULTS\n\n"
    )

    f.write(
        "Baseline HC3\n"
    )

    f.write(
        str(model_base.summary())
    )


    f.write(
        "\n\n\n"
        "Region Fixed Effects HC3\n"
    )

    f.write(
        str(model_fe.summary())
    )


    f.write(
        "\n\n\nCoefficient Comparison\n"
    )

    f.write(
        str(comparison)
    )


    f.write(
        "\n\n\nLeave-one-out\n"
    )

    f.write(
        str(loo)
    )


loo.to_csv(
    "outputs/tables/leave_one_out.csv",
    index=False
)


print("\n")
print("=" * 70)
print("DONE")
print("=" * 70)

print(
    "Saved:"
)

print(
    "outputs/tables/robustness_results.txt"
)

print(
    "outputs/tables/leave_one_out.csv"
)