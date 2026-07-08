"""
Specification Curve Analysis: GDP変数を整理し、全モデルの係数を可視化
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Specification Curve Analysis（GDP変数整理版）")
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
df['log_gdp_per_capita'] = np.log(df['gdp'] / df['population'])
df['log_population'] = np.log(df['population'] + 1)
df['log_co2_per_capita'] = np.log1p(df['co2_per_capita'])
df['total_co2'] = df['co2_per_capita'] * df['population']
df['log_total_co2'] = np.log(df['total_co2'] + 1)
df['energy_per_capita_scaled'] = df['energy_per_capita'] / 1000

# Muslim人口数（追加）
df['muslim_population'] = (df['Muslims'] / 100) * df['population']
df['log_muslim_population'] = np.log(df['muslim_population'] + 1)

print("✅ 変数作成完了")

# ========================================
# 3. 分析用データ（iso_code と Region を含める）
# ========================================
analysis_vars = [
    'iso_code', 'Region',
    'Muslims', 'log_muslim_population',  # ← log_muslim_population を追加！
    'log_gdp', 'log_gdp_per_capita', 'log_population',
    'energy_per_capita_scaled',
    'log_co2_per_capita', 'log_total_co2'
]
analysis_df = df[analysis_vars].dropna()
print(f"サンプル数: {len(analysis_df)}")

# 地域ダミー
region_dummies = pd.get_dummies(analysis_df['Region'], drop_first=True, dtype=int)

# ========================================
# 4. VIFチェック（修正版：GDP変数を同時に入れない）
# ========================================
print("\n" + "=" * 70)
print("VIFチェック（修正版）")
print("=" * 70)

# 仕様A: GDP per capita + Population
vif_vars_A = ['Muslims', 'log_gdp_per_capita', 'log_population', 'energy_per_capita_scaled']
vif_df_A = analysis_df[vif_vars_A]

print("\n仕様A (GDP per capita + Population):")
for i, col in enumerate(vif_df_A.columns):
    vif = sm.OLS(vif_df_A[col], sm.add_constant(vif_df_A.drop(columns=[col]))).fit()
    print(f"  {col}: {vif.rsquared / (1 - vif.rsquared):.2f}")

# 仕様B: GDP total + Population
vif_vars_B = ['Muslims', 'log_gdp', 'log_population', 'energy_per_capita_scaled']
vif_df_B = analysis_df[vif_vars_B]

print("\n仕様B (GDP total + Population):")
for i, col in enumerate(vif_df_B.columns):
    vif = sm.OLS(vif_df_B[col], sm.add_constant(vif_df_B.drop(columns=[col]))).fit()
    print(f"  {col}: {vif.rsquared / (1 - vif.rsquared):.2f}")

# ========================================
# 5. モデル定義（仕様ごとに整理）
# ========================================
y_pc = analysis_df['log_co2_per_capita']
y_total = analysis_df['log_total_co2']

# モデル仕様一覧
specs = []

# 仕様1: 比率のみ（ベースライン）
X = sm.add_constant(analysis_df[['Muslims']])
specs.append(("S1: Muslim % only", sm.OLS(y_pc, X).fit(cov_type='HC3'), y_pc))

# 仕様2: 比率 + GDP per capita
X = sm.add_constant(analysis_df[['Muslims', 'log_gdp_per_capita']])
specs.append(("S2: % + GDPpc", sm.OLS(y_pc, X).fit(cov_type='HC3'), y_pc))

# 仕様3: 比率 + GDP per capita + Energy
X = sm.add_constant(analysis_df[['Muslims', 'log_gdp_per_capita', 'energy_per_capita_scaled']])
specs.append(("S3: % + GDPpc + Energy", sm.OLS(y_pc, X).fit(cov_type='HC3'), y_pc))

# 仕様4: 比率 + GDP total + Population
X = sm.add_constant(analysis_df[['Muslims', 'log_gdp', 'log_population']])
specs.append(("S4: % + GDP_total + Pop", sm.OLS(y_pc, X).fit(cov_type='HC3'), y_pc))

# 仕様5: 比率 + GDP total + Population + Energy
X = sm.add_constant(analysis_df[['Muslims', 'log_gdp', 'log_population', 'energy_per_capita_scaled']])
specs.append(("S5: % + GDP_total + Pop + Energy", sm.OLS(y_pc, X).fit(cov_type='HC3'), y_pc))

# 仕様6: 比率 + GDP per capita + Energy + Region FE
X = pd.concat([
    analysis_df[['Muslims', 'log_gdp_per_capita', 'energy_per_capita_scaled']],
    region_dummies
], axis=1)
X = sm.add_constant(X)
specs.append(("S6: % + GDPpc + Energy + Region FE", sm.OLS(y_pc, X).fit(cov_type='HC3'), y_pc))

# 仕様7: Total CO2 + 比率 + GDP per capita + Energy
X = sm.add_constant(analysis_df[['Muslims', 'log_gdp_per_capita', 'energy_per_capita_scaled']])
specs.append(("S7: Total CO2 + % + GDPpc + Energy", sm.OLS(y_total, X).fit(cov_type='HC3'), y_total))

# 仕様8: Total CO2 + 比率 + GDP total + Population + Energy
X = sm.add_constant(analysis_df[['Muslims', 'log_gdp', 'log_population', 'energy_per_capita_scaled']])
specs.append(("S8: Total CO2 + % + GDP_total + Pop + Energy", sm.OLS(y_total, X).fit(cov_type='HC3'), y_total))

# 仕様9: Total CO2 + 人口数 + GDP total + Population + Energy
X = sm.add_constant(analysis_df[['log_muslim_population', 'log_gdp', 'log_population', 'energy_per_capita_scaled']])
specs.append(("S9: Total CO2 + Muslim_pop + GDP_total + Pop + Energy", sm.OLS(y_total, X).fit(cov_type='HC3'), y_total))

# 仕様10: Total CO2 + 比率 + GDP per capita + Energy + Region FE
X = pd.concat([
    analysis_df[['Muslims', 'log_gdp_per_capita', 'energy_per_capita_scaled']],
    region_dummies
], axis=1)
X = sm.add_constant(X)
specs.append(("S10: Total CO2 + % + GDPpc + Energy + Region FE", sm.OLS(y_total, X).fit(cov_type='HC3'), y_total))

# ========================================
# 6. 結果出力
# ========================================
print("\n" + "=" * 70)
print("Specification Curve Results")
print("=" * 70)
print(f"{'Model':<50} {'Muslims coef':>12} {'p-value':>10} {'R²':>8}")
print("-" * 85)

results = []
for name, model, y in specs:
    # Muslims または log_muslim_population のどちらかを取得
    if 'Muslims' in model.pvalues:
        coef = model.params['Muslims']
        p = model.pvalues['Muslims']
        ci = model.conf_int().loc['Muslims']
        var_name = 'Muslims'
    elif 'log_muslim_population' in model.pvalues:
        coef = model.params['log_muslim_population']
        p = model.pvalues['log_muslim_population']
        ci = model.conf_int().loc['log_muslim_population']
        var_name = 'Muslim_pop'
    else:
        coef = np.nan
        p = np.nan
        ci = (np.nan, np.nan)
        var_name = 'N/A'
    
    results.append({
        'Model': name,
        'Variable': var_name,
        'Coefficient': coef,
        'p_value': p,
        'CI_lower': ci[0],
        'CI_upper': ci[1],
        'R2': model.rsquared,
        'N': int(model.nobs)
    })
    print(f"{name:<50} {coef:>12.4f} {p:>10.4f} {model.rsquared:>8.4f}")

# ========================================
# 7. 結果をDataFrameに保存
# ========================================
results_df = pd.DataFrame(results)
print("\n" + "=" * 70)
print("結果サマリー")
print("=" * 70)
print(results_df[['Model', 'Variable', 'Coefficient', 'p_value', 'CI_lower', 'CI_upper', 'R2']].to_string(index=False))

# ========================================
# 8. CSVに保存（Figure作成用）
# ========================================
results_df.to_csv('outputs/tables/specification_curve_results.csv', index=False, encoding='utf-8')
print("\n✅ 保存: outputs/tables/specification_curve_results.csv")

# ========================================
# 9. テキスト保存
# ========================================
with open('outputs/tables/specification_curve_results.txt', 'w', encoding='utf-8') as f:
    f.write("Specification Curve Results\n")
    f.write("=" * 70 + "\n\n")
    f.write(results_df[['Model', 'Variable', 'Coefficient', 'p_value', 'CI_lower', 'CI_upper', 'R2']].to_string(index=False))
    f.write("\n\n")
    for name, model, y in specs:
        f.write(f"{name}\n")
        f.write("-" * 40 + "\n")
        f.write(str(model.summary()))
        f.write("\n\n")

print("✅ 保存: outputs/tables/specification_curve_results.txt")
print("=" * 70)