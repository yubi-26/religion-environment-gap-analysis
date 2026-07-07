"""
Muslim人口数 vs 割合の比較分析 + 石油輸出国ダミー
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Muslim人口数 vs 割合：比較分析（完全版）")
print("=" * 70)

df = pd.read_csv('data/processed/merged_epi_full_2020.csv')

# ========================================
# 1. 変数作成
# ========================================
df['log_gdp_per_capita'] = np.log(df['gdp'] / df['population'])
df['log_population'] = np.log(df['population'] + 1)
df['log_co2_per_capita'] = np.log1p(df['co2_per_capita'])
df['total_co2'] = df['co2_per_capita'] * df['population']
df['log_total_co2'] = np.log(df['total_co2'] + 1)
df['energy_per_capita_scaled'] = df['energy_per_capita'] / 1000

# Muslim人口数（割合 × 人口）
df['muslim_population'] = (df['Muslims'] / 100) * df['population']
df['log_muslim_population'] = np.log(df['muslim_population'] + 1)

# 石油輸出国ダミー
oil_exporters = [
    'SAU', 'IRN', 'IRQ', 'KWT', 'QAT', 'ARE',  # 湾岸諸国
    'OMN', 'BHR', 'LBY', 'DZA', 'NGA', 'AGO'    # その他OPEC
]
df['oil_exporter'] = df['iso_code'].isin(oil_exporters).astype(int)

print(f"データ形状: {df.shape}")
print(f"石油輸出国数: {df['oil_exporter'].sum()}")

# ========================================
# 2. 分析用データ
# ========================================
analysis_vars = [
    'iso_code', 'Region',
    'Muslims', 'log_muslim_population',
    'log_gdp_per_capita', 'log_population',
    'energy_per_capita_scaled',
    'log_total_co2', 'oil_exporter'
]
analysis_df = df[analysis_vars].dropna()
print(f"サンプル数: {len(analysis_df)}")

# 地域ダミー
region_dummies = pd.get_dummies(analysis_df['Region'], drop_first=True, dtype=int)

y = analysis_df['log_total_co2']

# ========================================
# 3. モデル1: Muslim割合（ベースライン）
# ========================================
print("\n" + "=" * 70)
print("モデル1: Muslim割合 → Total CO2")
print("=" * 70)
X1 = sm.add_constant(analysis_df[['Muslims']])
model1 = sm.OLS(y, X1).fit(cov_type='HC3')
print(model1.summary())

# ========================================
# 4. モデル2: Muslim人口数
# ========================================
print("\n" + "=" * 70)
print("モデル2: Muslim人口数 → Total CO2")
print("=" * 70)
X2 = sm.add_constant(analysis_df[['log_muslim_population']])
model2 = sm.OLS(y, X2).fit(cov_type='HC3')
print(model2.summary())

# ========================================
# 5. モデル3: Muslim割合 + GDP per capita + Energy
# ========================================
print("\n" + "=" * 70)
print("モデル3: Muslim割合 + GDP + Energy")
print("=" * 70)
X3 = sm.add_constant(analysis_df[['Muslims', 'log_gdp_per_capita', 'energy_per_capita_scaled']])
model3 = sm.OLS(y, X3).fit(cov_type='HC3')
print(model3.summary())

# ========================================
# 6. モデル4: Muslim人口数 + GDP per capita + Energy
# ========================================
print("\n" + "=" * 70)
print("モデル4: Muslim人口数 + GDP + Energy")
print("=" * 70)
X4 = sm.add_constant(analysis_df[['log_muslim_population', 'log_gdp_per_capita', 'energy_per_capita_scaled']])
model4 = sm.OLS(y, X4).fit(cov_type='HC3')
print(model4.summary())

# ========================================
# 7. モデル5: Muslim割合 + GDP + Energy + 石油輸出国
# ========================================
print("\n" + "=" * 70)
print("モデル5: Muslim割合 + GDP + Energy + 石油輸出国")
print("=" * 70)
X5 = sm.add_constant(analysis_df[['Muslims', 'log_gdp_per_capita', 'energy_per_capita_scaled', 'oil_exporter']])
model5 = sm.OLS(y, X5).fit(cov_type='HC3')
print(model5.summary())

# ========================================
# 8. モデル6: Muslim割合 + GDP + Energy + Region FE
# ========================================
print("\n" + "=" * 70)
print("モデル6: Muslim割合 + GDP + Energy + Region FE")
print("=" * 70)
X6 = pd.concat([
    analysis_df[['Muslims', 'log_gdp_per_capita', 'energy_per_capita_scaled']],
    region_dummies
], axis=1)
X6 = sm.add_constant(X6)
model6 = sm.OLS(y, X6).fit(cov_type='HC3')
print(model6.summary())

# ========================================
# 9. モデル7: Muslim人口数 + 総人口（人口規模効果と宗教構成効果を分離）
# ========================================
print("\n" + "=" * 70)
print("モデル7: Muslim人口数 + 総人口（人口効果を分離）")
print("=" * 70)
X7 = sm.add_constant(analysis_df[['log_muslim_population', 'log_population', 
                                   'log_gdp_per_capita', 'energy_per_capita_scaled']])
model7 = sm.OLS(y, X7).fit(cov_type='HC3')
print(model7.summary())

# ========================================
# 10. モデル8: Muslim割合 + 総人口（構成効果と人口規模効果を分離）
# ========================================
print("\n" + "=" * 70)
print("モデル8: Muslim割合 + 総人口（構成効果と人口規模効果を分離）")
print("=" * 70)
X8 = sm.add_constant(analysis_df[['Muslims', 'log_population', 
                                   'log_gdp_per_capita', 'energy_per_capita_scaled']])
model8 = sm.OLS(y, X8).fit(cov_type='HC3')
print(model8.summary())

# ========================================
# 11. モデル9: Muslim割合 + 石油輸出国 + Region FE
# ========================================
print("\n" + "=" * 70)
print("モデル9: Muslim割合 + 石油輸出国 + Region FE")
print("=" * 70)
X9 = pd.concat([
    analysis_df[['Muslims', 'log_gdp_per_capita', 'energy_per_capita_scaled', 'oil_exporter']],
    region_dummies
], axis=1)
X9 = sm.add_constant(X9)
model9 = sm.OLS(y, X9).fit(cov_type='HC3')
print(model9.summary())

# ========================================
# 12. モデル比較
# ========================================
print("\n" + "=" * 70)
print("モデル比較（Total CO2）")
print("=" * 70)

models = [
    ("Model 1: Muslim % only", model1),
    ("Model 2: Muslim pop only", model2),
    ("Model 3: % + GDP + Energy", model3),
    ("Model 4: Muslim pop + GDP + Energy", model4),
    ("Model 5: % + GDP + Energy + Oil", model5),
    ("Model 6: % + GDP + Energy + Region FE", model6),
    ("Model 7: Muslim pop + Pop + GDP + Energy", model7),
    ("Model 8: % + Pop + GDP + Energy", model8),
    ("Model 9: % + Oil + GDP + Energy + Region FE", model9),
]

for name, m in models:
    if m is not None:
        # Muslim関連のp値を取得
        p_values = {}
        for var in ['Muslims', 'log_muslim_population']:
            if var in m.pvalues:
                p_values[var] = m.pvalues[var]
        p_str = ", ".join([f"{k}={v:.4f}" for k, v in p_values.items()])
        print(f"{name}: R²={m.rsquared:.4f}, {p_str}")

# ========================================
# 13. 結果保存
# ========================================
with open('outputs/tables/muslim_population_analysis.txt', 'w', encoding='utf-8') as f:
    f.write("Muslim人口数 vs 割合：比較分析（完全版）\n")
    f.write("=" * 70 + "\n\n")
    
    for name, m in models:
        if m is not None:
            f.write(f"{name}\n")
            f.write("-" * 40 + "\n")
            f.write(str(m.summary()))
            f.write("\n\n")

print("\n✅ 保存: outputs/tables/muslim_population_analysis.txt")
print("=" * 70)