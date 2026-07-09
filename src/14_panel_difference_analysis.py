"""
Task 3: 疑似パネル分析（一階差分）
2010年と2020年のPewデータを使って、宗教比率の変化とCO2変化の関連を検証
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Task 3: 疑似パネル分析（一階差分）")
print("=" * 70)

# ========================================
# 1. Pewデータ読み込み
# ========================================
pew = pd.read_csv('data/raw/pew_religion_data.csv')

print("Pewデータ列名:", pew.columns.tolist())

# Level=1（国レベル）のみ抽出
pew_country = pew[pew['Level'] == 1].copy()

print(f"国レベルデータ数: {len(pew_country)}")
print(f"年: {pew_country['Year'].unique()}")

# 2010年と2020年のデータを分ける
pew_2010 = pew_country[pew_country['Year'] == 2010][['Country', 'Muslims']].rename(columns={'Muslims': 'muslims_2010'})
pew_2020 = pew_country[pew_country['Year'] == 2020][['Country', 'Muslims']].rename(columns={'Muslims': 'muslims_2020'})

# 結合
panel = pew_2010.merge(pew_2020, on='Country', how='inner')
panel['muslim_change'] = panel['muslims_2020'] - panel['muslims_2010']

print(f"パネルデータ（Pew）国数: {len(panel)}")
print(f"Muslim変化: 平均={panel['muslim_change'].mean():.2f}, "
      f"最大={panel['muslim_change'].max():.2f}, "
      f"最小={panel['muslim_change'].min():.2f}")

# ========================================
# 2. CO2データ読み込み
# ========================================
co2 = pd.read_csv('data/raw/owid-co2-data.csv')
co2['year'] = pd.to_numeric(co2['year'], errors='coerce')

co2_2010 = co2[co2['year'] == 2010][['country', 'co2_per_capita']].rename(columns={'co2_per_capita': 'co2_2010'})
co2_2020 = co2[co2['year'] == 2020][['country', 'co2_per_capita']].rename(columns={'co2_per_capita': 'co2_2020'})

print(f"CO2 2010データ数: {len(co2_2010)}")
print(f"CO2 2020データ数: {len(co2_2020)}")

# ========================================
# 3. パネルデータ作成
# ========================================
panel = panel.merge(co2_2010, left_on='Country', right_on='country', how='inner')
panel = panel.merge(co2_2020, left_on='Country', right_on='country', how='inner')

panel['co2_change'] = np.log(panel['co2_2020'] + 1e-6) - np.log(panel['co2_2010'] + 1e-6)
panel['log_co2_2020'] = np.log(panel['co2_2020'] + 1e-6)

print(f"最終パネルデータ国数: {len(panel)}")
print(panel[['Country', 'muslims_2010', 'muslims_2020', 'muslim_change', 'co2_change']].head())

# ========================================
# 4. 一階差分モデル（国固定効果相当）
# ========================================
X_fd = sm.add_constant(panel[['muslim_change']])
y_fd = panel['co2_change']
model_fd = sm.OLS(y_fd, X_fd, missing='drop').fit(cov_type='HC3')

print("\n=== 一階差分モデル ===")
print(model_fd.summary())

# ========================================
# 5. 横断面モデル（2020年のみ、比較用）
# ========================================
X_cross = sm.add_constant(panel[['muslims_2020']])
y_cross = panel['log_co2_2020']
model_cross = sm.OLS(y_cross, X_cross, missing='drop').fit(cov_type='HC3')

print("\n=== 横断面モデル（2020年のみ） ===")
print(model_cross.summary())

# ========================================
# 6. 結果の解釈
# ========================================
print("\n" + "=" * 70)
print("=== 結果の解釈 ===")
print("=" * 70)

coef_fd = model_fd.params.get('muslim_change', 0)
p_fd = model_fd.pvalues.get('muslim_change', 1)
coef_cross = model_cross.params.get('muslims_2020', 0)
p_cross = model_cross.pvalues.get('muslims_2020', 1)

print(f"\n一階差分モデル（国固定効果相当）:")
print(f"  Muslim変化の係数: {coef_fd:.4f}")
print(f"  p値: {p_fd:.4f}")
print(f"  R²: {model_fd.rsquared:.4f}")

print(f"\n横断面モデル（2020年のみ）:")
print(f"  Muslim比率の係数: {coef_cross:.4f}")
print(f"  p値: {p_cross:.4f}")
print(f"  R²: {model_cross.rsquared:.4f}")

print("\n" + "-" * 40)
print("解釈:")
print("- 一階差分: 同じ国でのMuslim変化とCO2変化の関連（国固有の要因を制御）")
print("- 横断面: 国ごとのMuslim比率とCO2排出量の関連（交絡の可能性あり）")
print("- 両者の差が、国固有の文化・地理・歴史的要因の交絡を示唆する")

if abs(coef_fd) < 0.01 and p_fd > 0.1:
    print("\n⚠️ 一階差分モデルでは有意な関連が確認できませんでした。")
    print("   これは宗教比率の10年変化が小さいため、検出力が低い可能性があります。")
    print("   この結果は「宗教構成の変化がCO2変化を説明しない」ことを示唆します。")

# ========================================
# 7. 結果保存
# ========================================
with open('outputs/tables/panel_difference_results.txt', 'w', encoding='utf-8') as f:
    f.write("Task 3: 疑似パネル分析結果\n")
    f.write("=" * 70 + "\n\n")
    f.write("一階差分モデル（国固定効果相当）\n")
    f.write(str(model_fd.summary()))
    f.write("\n\n")
    f.write("横断面モデル（2020年のみ）\n")
    f.write(str(model_cross.summary()))

print("\n✅ 保存: outputs/tables/panel_difference_results.txt")