import pandas as pd
import os


print("="*70)
print("MERGE ENVIRONMENTAL + INSTITUTION DATA")
print("="*70)


# ==========================
# Load data
# ==========================

base = pd.read_csv(
    "data/processed/merged_2020.csv"
)

epi = pd.read_csv(
    "data/raw/epi_2024_raw.csv"
)


print("Base:", base.shape)
print("EPI:", epi.shape)


# ==========================
# Select EPI variables
# ==========================

epi_cols = [
    "country",

    # Main environmental index
    "EPI.new",

    # Governance / institution related
    "RLI.new",
    "PRS.new",

    # Social indicators
    "SPI.new",
    "SHI.new",

    # Biodiversity
    "BDH.new",

    # Climate related
    "ECS.new",
    "GHN.new",

    # Pollution
    "AIR.new",
    "HPE.new",

    # Water
    "H2O.new",

    # Agriculture
    "AGR.new",

    # Gender already exists in base
]


# only existing columns
epi_cols = [
    c for c in epi_cols
    if c in epi.columns
]


print("\nSelected EPI columns:")
print(epi_cols)


epi_selected = epi[epi_cols].copy()


# Rename

rename_map = {
    "EPI.new":"epi_score",
    "RLI.new":"rule_of_law",
    "PRS.new":"political_rights",
    "SPI.new":"social_progress",
    "SHI.new":"sustainability_health",
    "BDH.new":"biodiversity",
    "ECS.new":"ecosystem_climate",
    "AIR.new":"air_quality",
    "H2O.new":"water_quality",
    "AGR.new":"agriculture",
}


epi_selected = epi_selected.rename(
    columns=rename_map
)


# ==========================
# Merge
# ==========================

merged = base.merge(
    epi_selected,
    left_on="country",
    right_on="country",
    how="left"
)


print("\nAfter merge:", merged.shape)


print("\nMissing:")
for c in rename_map.values():
    if c in merged.columns:
        print(
            c,
            merged[c].isna().sum()
        )


# ==========================
# Save
# ==========================

os.makedirs(
    "data/processed",
    exist_ok=True
)


output = (
    "data/processed/"
    "merged_epi_full_2020.csv"
)


merged.to_csv(
    output,
    index=False
)


print("\nSaved:")
print(output)