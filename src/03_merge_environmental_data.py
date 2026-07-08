"""
環境データ結合スクリプト
EPIデータをメインデータセットに結合する
"""

import pandas as pd
import numpy as np

print("=" * 60)
print("環境データ結合開始")
print("=" * 60)

# ========================================
# 1. データ読み込み
# ========================================
df = pd.read_csv('data/processed/merged_2020.csv')
epi = pd.read_csv('data/raw/epi_2024_raw.csv')

print(f"メインデータ: {df.shape}")
print(f"EPIデータ: {epi.shape}")

# ========================================
# 2. データ型を統一
# ========================================
df['iso_code'] = df['Countrycode'].astype(str).str.strip()
epi['iso_code'] = epi['code'].astype(str).str.strip()

# ========================================
# 3. EPI変数のマッピング（完全版：EPI.newを含む）
# ========================================
epi_vars = {
    # ⭐ 総合環境パフォーマンス（重要！）
    'EPI.new': 'epi_score',
    
    # Ecosystem Vitality
    'BDH.new': 'biodiversity_habitat',
    'ECS.new': 'forests',
    'AGR.new': 'agriculture',
    
    # Environmental Health
    'H2O.new': 'sanitation_water',
    'AIR.new': 'air_quality',
    
    # Biodiversity mechanisms
    'RLI.new': 'red_list_index',
    'SHI.new': 'species_habitat',
    'SPI.new': 'species_protection',
    
    # Pollution mechanism
    'PRS.new': 'pesticide_risk',
}

# ========================================
# 4. EPIデータから必要なカラムを抽出
# ========================================
epi_cols = ['iso_code'] + list(epi_vars.keys())
epi_subset = epi[epi_cols].copy()
epi_subset = epi_subset.rename(columns=epi_vars)

print(f"抽出したEPIカラム: {epi_subset.columns.tolist()}")

# ========================================
# 5. 結合
# ========================================
merged = df.merge(epi_subset, on='iso_code', how='left')
print(f"結合後: {merged.shape}")

# ========================================
# 6. 欠損状況確認
# ========================================
print("\n=== 欠損状況 ===")
for col in epi_vars.values():
    missing = merged[col].isnull().sum()
    print(f"{col}: {missing} 件欠損")

# ========================================
# 7. 保存
# ========================================
merged.to_csv('data/processed/merged_epi_full_2020.csv', index=False)
print("\n✅ 保存完了: data/processed/merged_epi_full_2020.csv")
print("=" * 60)