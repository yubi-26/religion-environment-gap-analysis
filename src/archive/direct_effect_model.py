"""
直接効果モデル
Muslims → CO2（GDP制御）
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

print("=" * 60)
print("直接効果モデル: Muslims → CO2")
print("=" * 60)

# データ読み込み
df = pd.read_csv('data/processed/merged_epi_full_2020.csv')

# 変数定義
X_var = 'Muslims'
Y_var = 'co2_per_capita'
control_vars = ['log_gdp']

# データ準備
df['log_gdp'] = np.log(df['gdp'])
df['log_co2'] = np.log1p(df['co2_per_capita'])

# 分析用データ
analysis_df = df[[X_var, 'log_co2'] + control_vars].dropna()
print(f"サンプル数: {len(analysis_df)}")

# モデル1: Muslims → CO2（制御なし）
X1 = sm.add_constant(analysis_df[[X_var]])
y = analysis_df['log_co2']
model1 = sm.OLS(y, X1).fit(cov_type='HC3')
print("\n=== モデル1: Muslims → CO2（制御なし） ===")
print(model1.summary())

# モデル2: Muslims + GDP → CO2（GDP制御）
X2 = sm.add_constant(analysis_df[[X_var] + control_vars])
model2 = sm.OLS(y, X2).fit(cov_type='HC3')
print("\n=== モデル2: Muslims + GDP → CO2（GDP制御） ===")
print(model2.summary())

# 結果保存
with open('outputs/tables/direct_effect_model.txt', 'w', encoding='utf-8') as f:
    f.write("直接効果モデル\n")
    f.write("=" * 60 + "\n")
    f.write("モデル1: Muslims → CO2（制御なし）\n")
    f.write(str(model1.summary()))
    f.write("\n\n")
    f.write("モデル2: Muslims + GDP → CO2（GDP制御）\n")
    f.write(str(model2.summary()))

print("\n✅ 保存: outputs/tables/direct_effect_model.txt")