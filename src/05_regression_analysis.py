"""
回帰分析スクリプト
Religion-Environment Gap Analysis
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from statsmodels.stats.outliers_influence import variance_inflation_factor

print("=" * 60)
print("回帰分析開始")
print("=" * 60)

# ========================================
# 1. データ読み込み
# ========================================
df = pd.read_csv('data/processed/merged_2020.csv')
print(f"データ形状: {df.shape}")

# カラム名を確認
print(f"カラム一覧: {df.columns.tolist()}")

# ========================================
# 2. 変数定義（実際のカラム名に合わせる）
# ========================================
religion_cols = ['Muslims', 'Christians', 'Buddhists', 'Hindus']

# 対数変換
df['log_gdp'] = np.log(df['gdp'])
df['log_co2'] = np.log(df['co2_per_capita'].replace(0, np.nan))

# ========================================
# 3. 欠損値処理
# ========================================
analysis_cols = religion_cols + ['log_gdp', 'log_co2']
df_clean = df[analysis_cols].dropna()
print(f"欠損除去後のデータ数: {len(df_clean)}")

# ========================================
# 4. OLS回帰
# ========================================
X = sm.add_constant(df_clean[religion_cols + ['log_gdp']])
y = df_clean['log_co2']

model = sm.OLS(y, X).fit(cov_type='HC3')

print("\n" + "=" * 60)
print("OLS回帰結果")
print("=" * 60)
print(model.summary())

# ========================================
# 5. VIF（多重共線性チェック）
# ========================================
print("\n" + "=" * 60)
print("VIF（多重共線性チェック）")
print("=" * 60)

vif_data = sm.add_constant(df_clean[religion_cols + ['log_gdp']]).dropna()
for i, col in enumerate(vif_data.columns):
    if col != 'const':
        vif = variance_inflation_factor(vif_data.values, i)
        print(f"{col}: {vif:.2f}")

# ========================================
# 6. 結果要約
# ========================================
print("\n" + "=" * 60)
print("結果要約")
print("=" * 60)
print(f"R-squared: {model.rsquared:.4f}")
print(f"Adj. R-squared: {model.rsquared_adj:.4f}")
print(f"サンプル数: {int(model.nobs)}")

print("\n【各変数の係数と有意性】")
for var in ['Muslims', 'Christians', 'Buddhists', 'Hindus', 'log_gdp']:
    if var in model.params.index:
        p_val = model.pvalues[var]
        sig = "***" if p_val < 0.001 else "**" if p_val < 0.01 else "*" if p_val < 0.05 else ""
        print(f"{var}: 係数={model.params[var]:.4f}, p値={p_val:.4f} {sig}")

print("\n" + "=" * 60)
print("回帰分析完了")
print("=" * 60)

# 結果をファイルに保存
with open('outputs/tables/regression_full_results.txt', 'w', encoding='utf-8') as f:
    f.write("=" * 60 + "\n")
    f.write("OLS回帰結果\n")
    f.write("=" * 60 + "\n")
    f.write(model.summary().as_text())
    f.write("\n\n" + "=" * 60 + "\n")
    f.write("VIF（多重共線性チェック）\n")
    f.write("=" * 60 + "\n")
    for i, col in enumerate(vif_data.columns):
        if col != 'const':
            f.write(f"{col}: {variance_inflation_factor(vif_data.values, i):.2f}\n")

print("\n✅ 結果を保存: outputs/tables/regression_full_results.txt")