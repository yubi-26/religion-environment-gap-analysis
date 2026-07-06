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

# 数値カラムに変換
df['Muslims'] = df['Muslims'].astype(float)

# 欠損値の確認
print("\n=== 欠損値確認 ===")
print(f"Muslims 欠損: {df['Muslims'].isnull().sum()}")
print(f"co2_per_capita 欠損: {df['co2_per_capita'].isnull().sum()}")
print(f"gdp 欠損: {df['gdp'].isnull().sum()}")

# 分析に使うカラムを選択
analysis_df = df[['Muslims', 'co2_per_capita', 'gdp']].copy()

# 欠損値を含む行を削除（リストワイズ削除）
analysis_df = analysis_df.dropna()

print(f"\n欠損値除去後のデータ数: {len(analysis_df)}")

# 1. 単回帰分析：Muslims → CO2_per_capita
X1 = sm.add_constant(analysis_df['Muslims'])
y = analysis_df['co2_per_capita']

model1 = sm.OLS(y, X1).fit()
print("\n=== モデル1: Muslims → CO2_per_capita ===")
print(model1.summary())

# 2. 重回帰分析：Muslims + log(GDP) → CO2_per_capita
analysis_df['log_gdp'] = np.log(analysis_df['gdp'])

X2 = sm.add_constant(analysis_df[['Muslims', 'log_gdp']])
model2 = sm.OLS(y, X2).fit()
print("\n=== モデル2: Muslims + log(GDP) → CO2_per_capita ===")
print(model2.summary())

# 3. 回帰式を使った予測
print("\n=== モデル2の解釈 ===")
print(f"定数項: {model2.params['const']:.4f}")
print(f"Muslims係数: {model2.params['Muslims']:.4f} (p={model2.pvalues['Muslims']:.4f})")
print(f"log(GDP)係数: {model2.params['log_gdp']:.4f} (p={model2.pvalues['log_gdp']:.4f})")
print(f"R²: {model2.rsquared:.4f}")

# 散布図の作成
fig, axes = plt.subplots(1, 3, figsize=(15, 5))

# Muslim vs CO2
axes[0].scatter(analysis_df['Muslims'], analysis_df['co2_per_capita'], alpha=0.6)
axes[0].set_xlabel('Muslim Population (%)')
axes[0].set_ylabel('CO2 per capita (tons)')
axes[0].set_title('Muslims vs CO2')

# log(GDP) vs CO2
axes[1].scatter(analysis_df['log_gdp'], analysis_df['co2_per_capita'], alpha=0.6)
axes[1].set_xlabel('log(GDP)')
axes[1].set_ylabel('CO2 per capita (tons)')
axes[1].set_title('log(GDP) vs CO2')

# Muslim vs log(GDP)
axes[2].scatter(analysis_df['Muslims'], analysis_df['log_gdp'], alpha=0.6)
axes[2].set_xlabel('Muslim Population (%)')
axes[2].set_ylabel('log(GDP)')
axes[2].set_title('Muslims vs log(GDP)')

plt.tight_layout()
plt.savefig('outputs/figures/regression_scatter.png', dpi=300)
print("✅ 散布図を保存しました: outputs/figures/regression_scatter.png")

# 結果をテキストで保存（UTF-8指定）
with open('outputs/tables/regression_results.txt', 'w', encoding='utf-8') as f:
    f.write("=== 回帰分析結果 ===\n\n")
    f.write(f"分析対象国数: {len(analysis_df)}\n\n")
    f.write("モデル1: Muslims → CO2_per_capita\n")
    f.write(str(model1.summary()))
    f.write("\n\n")
    f.write("モデル2: Muslims + log(GDP) → CO2_per_capita\n")
    f.write(str(model2.summary()))

print("✅ 結果を保存しました: outputs/tables/regression_results.txt")