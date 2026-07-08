import pandas as pd
import numpy as np

print("=== データ処理開始 ===")

# 1. データ読み込み
pew = pd.read_csv('data/raw/pew_religion_data.csv')
co2 = pd.read_csv('data/raw/owid-co2-data.csv')

print(f"Pewデータ: {pew.shape}")
print(f"CO2データ: {co2.shape}")

# 2. 2020年データに絞る
pew_2020 = pew[pew['Year'] == 2020]
co2_2020 = co2[co2['year'] == 2020]

print(f"Pew 2020: {len(pew_2020)} rows")
print(f"CO2 2020: {len(co2_2020)} rows")

# 3. データ結合
merged = pew_2020.merge(co2_2020, left_on='Country', right_on='country', how='inner')

print(f"結合後: {len(merged)} rows, {merged['Country'].nunique()} countries")

# 4. 保存
merged.to_csv('data/processed/merged_2020.csv', index=False)
print("✅ 結合データを保存しました: data/processed/merged_2020.csv")

# 5. 基本統計
print("\n=== 基本統計 ===")
print(merged[['Muslims', 'co2_per_capita']].describe())

# 6. 相関
corr = merged[['Muslims', 'co2_per_capita']].corr()
print(f"\nイスラム比率 vs CO2排出量 相関: {corr.iloc[0,1]:.3f}")