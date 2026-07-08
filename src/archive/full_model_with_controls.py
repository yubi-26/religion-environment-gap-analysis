"""
完全モデル：Muslims + 交絡変数 → CO2
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("完全モデル: Muslims + 交絡変数 → CO2")
print("=" * 70)

# データ読み込み
df = pd.read_csv('data/processed/merged_epi_full_2020.csv')
print(f"データ形状: {df.shape}")

# ========================================
# log変換
# ========================================
df['log_gdp'] = np.log(df['gdp'])
df['log_co2'] = np.log1p(df['co2_per_capita'])
df['log_population'] = np.log(df['population'] + 1)  # 安全のため+1

# ========================================
# 変数定義
# ========================================
control_vars = ['log_gdp', 'energy_per_capita', 'log_population']
available_vars = [v for v in control_vars if v in df.columns]
print(f"\n利用可能な制御変数: {available_vars}")

# 分析用データ
analysis_vars = ['Muslims', 'log_co2'] + available_vars
analysis_df = df[analysis_vars].dropna()
print(f"サンプル数: {len(analysis_df)}")

# ========================================
# モデル1: Muslimsのみ
# ========================================
X1 = sm.add_constant(analysis_df[['Muslims']])
y = analysis_df['log_co2']
model1 = sm.OLS(y, X1).fit(cov_type='HC3')
print("\n=== モデル1: Muslimsのみ ===")
print(model1.summary())

# ========================================
# モデル2: Muslims + GDP
# ========================================
X2 = sm.add_constant(analysis_df[['Muslims', 'log_gdp']])
model2 = sm.OLS(y, X2).fit(cov_type='HC3')
print("\n=== モデル2: Muslims + GDP ===")
print(model2.summary())

# ========================================
# モデル3: Muslims + GDP + Population
# ========================================
X3 = sm.add_constant(analysis_df[['Muslims', 'log_gdp', 'log_population']])
model3 = sm.OLS(y, X3).fit(cov_type='HC3')
print("\n=== モデル3: Muslims + GDP + Population ===")
print(model3.summary())

# ========================================
# モデル4: Muslims + GDP + Population + Energy
# ========================================
if 'energy_per_capita' in analysis_df.columns:
    X4 = sm.add_constant(analysis_df[['Muslims', 'log_gdp', 'log_population', 'energy_per_capita']])
    model4 = sm.OLS(y, X4).fit(cov_type='HC3')
    print("\n=== モデル4: Muslims + GDP + Population + Energy ===")
    print(model4.summary())
else:
    print("\n⚠️ energy_per_capita が利用できません。モデル4をスキップします。")
    model4 = None

# ========================================
# モデル比較
# ========================================
print("\n" + "=" * 70)
print("モデル比較")
print("=" * 70)
print(f"モデル1 (Muslims only): R² = {model1.rsquared:.4f}, Muslims p = {model1.pvalues['Muslims']:.4f}")
print(f"モデル2 (+GDP): R² = {model2.rsquared:.4f}, Muslims p = {model2.pvalues['Muslims']:.4f}")
print(f"モデル3 (+GDP+Pop): R² = {model3.rsquared:.4f}, Muslims p = {model3.pvalues['Muslims']:.4f}")
if model4:
    print(f"モデル4 (+GDP+Pop+Energy): R² = {model4.rsquared:.4f}, Muslims p = {model4.pvalues['Muslims']:.4f}")

# ========================================
# 結果保存（UTF-8指定）
# ========================================
with open('outputs/tables/full_model_results.txt', 'w', encoding='utf-8') as f:
    f.write("完全モデル結果\n")
    f.write("=" * 70 + "\n")
    f.write("モデル1: Muslimsのみ\n")
    f.write(str(model1.summary()))
    f.write("\n\n")
    f.write("モデル2: Muslims + GDP\n")
    f.write(str(model2.summary()))
    f.write("\n\n")
    f.write("モデル3: Muslims + GDP + Population\n")
    f.write(str(model3.summary()))
    if model4:
        f.write("\n\n")
        f.write("モデル4: Muslims + GDP + Population + Energy\n")
        f.write(str(model4.summary()))

print("\n✅ 保存: outputs/tables/full_model_results.txt")