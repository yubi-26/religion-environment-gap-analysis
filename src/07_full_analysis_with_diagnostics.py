"""
完全分析：VIFチェック + Total CO2 + 地域固定効果（完全修正版）
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("完全分析：VIFチェック + Total CO2 + 地域固定効果（完全修正版）")
print("=" * 70)

# ========================================
# 1. データ読み込み
# ========================================
df = pd.read_csv('data/processed/merged_epi_full_2020.csv')
print(f"データ形状: {df.shape}")

# ========================================
# 2. 変数作成
# ========================================
df['log_gdp'] = np.log(df['gdp'])
df['log_gdp_per_capita'] = np.log(df['gdp'] / df['population'])  # GDP per capita 追加
df['log_population'] = np.log(df['population'] + 1)
df['log_co2_per_capita'] = np.log1p(df['co2_per_capita'])
df['total_co2'] = df['co2_per_capita'] * df['population']
df['log_total_co2'] = np.log(df['total_co2'] + 1)
df['energy_per_capita_scaled'] = df['energy_per_capita'] / 1000

print("✅ 変数作成完了")

# ========================================
# 3. 分析用データ（iso_code を必ず含める）
# ========================================
analysis_vars = [
    'iso_code',
    'Muslims',
    'log_gdp',
    'log_gdp_per_capita',
    'log_population',
    'energy_per_capita_scaled',
    'log_co2_per_capita',
    'log_total_co2'
]
analysis_df = df[analysis_vars].dropna()
print(f"サンプル数: {len(analysis_df)}")

# ========================================
# 4. VIFチェック
# ========================================
print("\n" + "=" * 70)
print("VIFチェック（多重共線性）")
print("=" * 70)

vif_vars = ['Muslims', 'log_gdp', 'log_gdp_per_capita', 'log_population', 'energy_per_capita_scaled']
vif_df = analysis_df[vif_vars]

vif_results = []
for i, col in enumerate(vif_df.columns):
    vif = variance_inflation_factor(vif_df.values, i)
    vif_results.append((col, vif))
    print(f"{col}: {vif:.2f}")

high_vif = [col for col, vif in vif_results if vif > 10]
if high_vif:
    print(f"\n⚠️ VIF > 10 の変数: {high_vif}")
else:
    print("\n✅ すべてのVIFが10未満です。")

# ========================================
# 5. モデル1: CO2 per capita（ベースライン）
# ========================================
print("\n" + "=" * 70)
print("モデル1: CO2 per capita（ベースライン）")
print("=" * 70)

X1 = sm.add_constant(analysis_df[['Muslims']])
y1 = analysis_df['log_co2_per_capita']
model1 = sm.OLS(y1, X1).fit(cov_type='HC3')
print(model1.summary())

# ========================================
# 6. モデル2: CO2 per capita + GDP
# ========================================
print("\n" + "=" * 70)
print("モデル2: CO2 per capita + GDP")
print("=" * 70)

X2 = sm.add_constant(analysis_df[['Muslims', 'log_gdp']])
model2 = sm.OLS(y1, X2).fit(cov_type='HC3')
print(model2.summary())

# ========================================
# 7. モデル3: CO2 per capita + GDP + Population
# ========================================
print("\n" + "=" * 70)
print("モデル3: CO2 per capita + GDP + Population")
print("=" * 70)

X3 = sm.add_constant(analysis_df[['Muslims', 'log_gdp', 'log_population']])
model3 = sm.OLS(y1, X3).fit(cov_type='HC3')
print(model3.summary())

# ========================================
# 8. モデル4: CO2 per capita + GDP + Population + Energy
# ========================================
print("\n" + "=" * 70)
print("モデル4: CO2 per capita + GDP + Population + Energy")
print("=" * 70)

X4 = sm.add_constant(analysis_df[['Muslims', 'log_gdp', 'log_population', 'energy_per_capita_scaled']])
model4 = sm.OLS(y1, X4).fit(cov_type='HC3')
print(model4.summary())

# ========================================
# 9. モデル5: Total CO2（人口問題を回避）
# ========================================
print("\n" + "=" * 70)
print("モデル5: Total CO2（人口を説明変数から除外）")
print("=" * 70)

y2 = analysis_df['log_total_co2']
X5 = sm.add_constant(analysis_df[['Muslims', 'log_gdp', 'energy_per_capita_scaled']])
model5 = sm.OLS(y2, X5).fit(cov_type='HC3')
print(model5.summary())

# ========================================
# 10. モデル6: 地域固定効果（GDP total）
# ========================================
print("\n" + "=" * 70)
print("モデル6: 地域固定効果（GDP total）")
print("=" * 70)

# 地域データを結合
region_df = df[['iso_code', 'Region']].drop_duplicates()
analysis_with_region = analysis_df.merge(region_df, on='iso_code', how='left')
analysis_with_region = analysis_with_region.dropna(subset=['Region'])
print(f"地域情報ありサンプル数: {len(analysis_with_region)}")

# 地域ダミー（dtype=int を明示）
region_dummies = pd.get_dummies(analysis_with_region['Region'], drop_first=True, dtype=int)

# 説明変数
X6 = pd.concat([
    analysis_with_region[['Muslims', 'log_gdp', 'log_population', 'energy_per_capita_scaled']],
    region_dummies
], axis=1)

# 全ての列を float に変換
for col in X6.columns:
    X6[col] = X6[col].astype(float)

X6 = sm.add_constant(X6)
y6 = analysis_with_region['log_co2_per_capita'].astype(float)

# 欠損を削除
valid_idx = ~(X6.isna().any(axis=1) | y6.isna())
X6_clean = X6[valid_idx].astype(float)
y6_clean = y6[valid_idx].astype(float)

print(f"最終サンプル数: {len(X6_clean)}")
print(f"地域カテゴリ数: {len(region_dummies.columns) + 1}")

model6 = sm.OLS(y6_clean, X6_clean).fit(cov_type='HC3')
print(model6.summary())

# ========================================
# 11. モデル7: GDP per capita + 地域固定効果
# ========================================
print("\n" + "=" * 70)
print("モデル7: GDP per capita + 地域固定効果")
print("=" * 70)

X7 = pd.concat([
    analysis_with_region[['Muslims', 'log_gdp_per_capita', 'log_population', 'energy_per_capita_scaled']],
    region_dummies
], axis=1)

for col in X7.columns:
    X7[col] = X7[col].astype(float)

X7 = sm.add_constant(X7)
y7 = analysis_with_region['log_co2_per_capita'].astype(float)

valid_idx7 = ~(X7.isna().any(axis=1) | y7.isna())
X7_clean = X7[valid_idx7].astype(float)
y7_clean = y7[valid_idx7].astype(float)

print(f"最終サンプル数: {len(X7_clean)}")

model7 = sm.OLS(y7_clean, X7_clean).fit(cov_type='HC3')
print(model7.summary())

# ========================================
# 12. モデル比較
# ========================================
print("\n" + "=" * 70)
print("モデル比較")
print("=" * 70)

models = [
    ("Model 1: Muslims only", model1),
    ("Model 2: + GDP", model2),
    ("Model 3: + GDP + Pop", model3),
    ("Model 4: + GDP + Pop + Energy", model4),
    ("Model 5: Total CO2 (no Pop)", model5),
    ("Model 6: Region FE (GDP total)", model6),
    ("Model 7: Region FE (GDP per capita)", model7),
]

for name, m in models:
    if m is not None:
        muslims_p = m.pvalues.get('Muslims', np.nan)
        print(f"{name}: R²={m.rsquared:.4f}, Muslims p={muslims_p:.4f}")

# ========================================
# 13. 結果保存
# ========================================
with open('outputs/tables/full_analysis_results_final.txt', 'w', encoding='utf-8') as f:
    f.write("完全分析結果（最終版）\n")
    f.write("=" * 70 + "\n\n")
    
    f.write("VIFチェック\n")
    f.write("-" * 40 + "\n")
    for col, vif in vif_results:
        f.write(f"{col}: {vif:.2f}\n")
    f.write("\n\n")
    
    for name, m in models:
        if m is not None:
            f.write(f"{name}\n")
            f.write("-" * 40 + "\n")
            f.write(str(m.summary()))
            f.write("\n\n")

print("\n✅ 保存: outputs/tables/full_analysis_results_final.txt")
print("=" * 70)