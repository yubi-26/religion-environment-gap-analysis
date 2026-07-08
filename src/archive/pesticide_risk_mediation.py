"""
殺虫剤リスク媒介分析（Pesticide Risk Mediation）
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm

print("=" * 60)
print("殺虫剤リスク媒介分析（Pesticide Risk Mediation）")
print("=" * 60)

df = pd.read_csv('data/processed/merged_epi_full_2020.csv')
print(f"データ形状: {df.shape}")

# 変数定義
mediator = 'pesticide_risk'
X_var = 'Muslims'
Y_var = 'epi_score'
control_vars = ['log_gdp']

print(f"媒介変数: {mediator}（殺虫剤リスク）")

df['log_gdp'] = np.log(df['gdp'])

analysis_cols = [X_var, mediator, Y_var] + control_vars
analysis_df = df[analysis_cols].dropna()
print(f"有効サンプル数: {len(analysis_df)}")

# Path A: X → M
X_A = sm.add_constant(analysis_df[[X_var] + control_vars])
y_A = analysis_df[mediator]
model_A = sm.OLS(y_A, X_A).fit(cov_type='HC3')

# Path B: M → Y
X_B = sm.add_constant(analysis_df[[mediator] + control_vars])
y_B = analysis_df[Y_var]
model_B = sm.OLS(y_B, X_B).fit(cov_type='HC3')

# Path C': X + M → Y
X_C = sm.add_constant(analysis_df[[X_var, mediator] + control_vars])
y_C = analysis_df[Y_var]
model_C = sm.OLS(y_C, X_C).fit(cov_type='HC3')

print(f"\nPath A (Muslims → 殺虫剤リスク):")
print(f"  係数: {model_A.params[X_var]:.4f}, p値: {model_A.pvalues[X_var]:.4f}")

print(f"\nPath B (殺虫剤リスク → EPI):")
print(f"  係数: {model_B.params[mediator]:.4f}, p値: {model_B.pvalues[mediator]:.4f}")

print(f"\nPath C' (Muslims → EPI, 制御後):")
print(f"  係数: {model_C.params[X_var]:.4f}, p値: {model_C.pvalues[X_var]:.4f}")

with open('outputs/tables/pesticide_risk_mediation.txt', 'w') as f:
    f.write("殺虫剤リスク媒介分析結果\n")
    f.write("=" * 60 + "\n")
    f.write(f"Path A: {model_A.params[X_var]:.4f} (p={model_A.pvalues[X_var]:.4f})\n")
    f.write(f"Path B: {model_B.params[mediator]:.4f} (p={model_B.pvalues[mediator]:.4f})\n")
    f.write(f"Path C': {model_C.params[X_var]:.4f} (p={model_C.pvalues[X_var]:.4f})\n")

print("\n✅ 保存: outputs/tables/pesticide_risk_mediation.txt")