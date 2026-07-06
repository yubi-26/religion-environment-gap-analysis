import pandas as pd
import numpy as np
import statsmodels.api as sm
import statsmodels.formula.api as smf
import matplotlib.pyplot as plt
import seaborn as sns

print("=== 回帰分析開始 ===")

# データ読み込み
df = pd.read_csv('data/processed/merged_2020.csv')

print(f"データ形状: {df.shape}")
print(f"カラム: {df.columns.tolist()}")

# 数値カラムに変換（%が文字列の場合があるため）
df['Muslims'] = df['Muslims'].astype(float)

# 1. 単回帰分析：Muslims → CO2_per_capita
X = sm.add_constant(df['Muslims'])
y = df['co2_per_capita']

model1 = sm.OLS(y, X).fit()
print("\n=== モデル1: Muslims → CO2_per_capita ===")
print(model1.summary())

# 2. 重回帰分析：Muslims + GDP → CO2_per_capita
# GDPの対数変換（外れ値対策）
df['log_gdp'] = np.log(df['gdp'])

X2 = sm.add_constant(df[['Muslims', 'log_gdp']])
model2 = sm.OLS(y, X2).fit()
print("\n=== モデル2: Muslims + log(GDP) → CO2_per_capita ===")
print(model2.summary())

# 3. 重回帰分析：Muslims + GDP + Population → CO2_per_capita
X3 = sm.add_constant(df[['Muslims', 'log_gdp', 'Population']])
model3 = sm.OLS(y, X3).fit()
print("\n=== モデル3: Muslims + log(GDP) + Population → CO2_per_capita ===")
print(model3.summary())

# 散布図の作成
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Muslim vs CO2
axes[0].scatter(df['Muslims'], df['co2_per_capita'], alpha=0.6)
axes[0].set_xlabel('Muslim Population (%)')
axes[0].set_ylabel('CO2 per capita (tons)')
axes[0].set_title('Muslims vs CO2')

# log(GDP) vs CO2
axes[1].scatter(df['log_gdp'], df['co2_per_capita'], alpha=0.6)
axes[1].set_xlabel('log(GDP)')
axes[1].set_ylabel('CO2 per capita (tons)')
axes[1].set_title('log(GDP) vs CO2')

# Muslim vs log(GDP)
axes[2].scatter(df['Muslims'], df['log_gdp'], alpha=0.6)
axes[2].set_xlabel('Muslim Population (%)')
axes[2].set_ylabel('log(GDP)')
axes[2].set_title('Muslims vs log(GDP)')

plt.tight_layout()
plt.savefig('outputs/figures/regression_scatter.png', dpi=300)
print("✅ 散布図を保存しました: outputs/figures/regression_scatter.png")

# 結果をテキストで保存
with open('outputs/tables/regression_results.txt', 'w') as f:
    f.write("=== 回帰分析結果 ===\n\n")
    f.write("モデル1: Muslims → CO2_per_capita\n")
    f.write(str(model1.summary()))
    f.write("\n\n")
    f.write("モデル2: Muslims + log(GDP) → CO2_per_capita\n")
    f.write(str(model2.summary()))
    f.write("\n\n")
    f.write("モデル3: Muslims + log(GDP) + Population → CO2_per_capita\n")
    f.write(str(model3.summary()))

print("✅ 結果を保存しました: outputs/tables/regression_results.txt")