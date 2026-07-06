"""
論文用Figure作成スクリプト
- Figure 1: First Stage (操作変数の関連性)
- Figure 2: OLS vs IV 係数比較
- Figure 3: 頑健性チェック
"""

import numpy as np
import pandas as pd
import matplotlib.pyplot as plt
import seaborn as sns
from linearmodels.iv import IV2SLS
from statsmodels.api import OLS, add_constant
import os

print("=" * 60)
print("論文用Figureを作成中...")
print("=" * 60)

# ================================
# 1. データ読み込み
# ================================
df = pd.read_csv('data/processed/merged_2020.csv')

# 地理データ（簡易）
latitude_dict = {
    'Afghanistan': 33.94, 'Albania': 41.15, 'Algeria': 28.03,
    'Germany': 51.17, 'France': 46.60, 'United States': 37.77,
    'Saudi Arabia': 23.89, 'Turkey': 39.03, 'India': 20.59,
    'China': 35.86, 'Japan': 36.20, 'Brazil': -14.24,
    'Nigeria': 9.08, 'Egypt': 26.82
}

longitude_dict = {
    'Afghanistan': 69.21, 'Albania': 19.82, 'Algeria': 3.09,
    'Germany': 10.45, 'France': 2.21, 'United States': -100.00,
    'Saudi Arabia': 45.08, 'Turkey': 35.24, 'India': 78.96,
    'China': 104.20, 'Japan': 138.25, 'Brazil': -51.93,
    'Nigeria': 8.68, 'Egypt': 30.80
}

def haversine(lat1, lon1, lat2, lon2):
    from math import radians, sin, cos, sqrt, atan2
    R = 6371
    lat1, lon1, lat2, lon2 = map(radians, [lat1, lon1, lat2, lon2])
    dlat = lat2 - lat1
    dlon = lon2 - lon1
    a = sin(dlat/2)**2 + cos(lat1)*cos(lat2)*sin(dlon/2)**2
    return 2 * R * atan2(sqrt(a), sqrt(1-a))

mecca_lat, mecca_lon = 21.3891, 39.8579

df['latitude'] = df['Country'].map(latitude_dict)
df['longitude'] = df['Country'].map(longitude_dict)

df['distance_to_mecca'] = df.apply(
    lambda r: haversine(r['latitude'], r['longitude'], mecca_lat, mecca_lon)/1000
    if pd.notnull(r['latitude']) else np.nan,
    axis=1
)

df = df.dropna(subset=['Muslims', 'gdp', 'co2_per_capita', 'latitude', 'distance_to_mecca'])
df['log_gdp'] = np.log(df['gdp'])
df['log_co2'] = np.log(df['co2_per_capita'] + 1e-6)

print(f"データ数: {len(df)}")

# 出力フォルダ作成
os.makedirs('outputs/figures', exist_ok=True)

# ================================
# Figure 1: First Stage (操作変数の関連性)
# ================================
fig, axes = plt.subplots(1, 2, figsize=(12, 5))

# 緯度 vs Muslims
axes[0].scatter(df['latitude'], df['Muslims'], alpha=0.6, s=50)
axes[0].set_xlabel('Latitude (degrees)', fontsize=12)
axes[0].set_ylabel('Muslim Population (%)', fontsize=12)
axes[0].set_title('(a) Latitude → Muslim Share', fontsize=14)
axes[0].grid(alpha=0.3)

# メッカ距離 vs Muslims
axes[1].scatter(df['distance_to_mecca'], df['Muslims'], alpha=0.6, s=50)
axes[1].set_xlabel('Distance from Mecca (thousand km)', fontsize=12)
axes[1].set_ylabel('Muslim Population (%)', fontsize=12)
axes[1].set_title('(b) Distance from Mecca → Muslim Share', fontsize=14)
axes[1].grid(alpha=0.3)

plt.tight_layout()
plt.savefig('outputs/figures/figure1_first_stage.png', dpi=300, bbox_inches='tight')
print("✅ Figure 1: outputs/figures/figure1_first_stage.png")

# ================================
# Figure 2: OLS vs IV 係数比較
# ================================
# OLS
ols = OLS(df['log_co2'], add_constant(df[['Muslims', 'log_gdp']])).fit()

# IV
model = IV2SLS(
    dependent=df['log_co2'],
    exog=df[['log_gdp']],
    endog=df['Muslims'],
    instruments=df[['latitude', 'distance_to_mecca']]
).fit(cov_type='robust')

# 係数と標準誤差を取得
coef_ols = ols.params['Muslims']
se_ols = ols.bse['Muslims']

# IVの係数と標準誤差（安全に取得）
coef_iv = model.params.get('Muslims', np.nan)
se_iv = model.std_errors.get('Muslims', np.nan)

plt.figure(figsize=(8, 5))
labels = ['OLS', 'IV (2SLS)']
coefs = [coef_ols, coef_iv]
errors = [se_ols, se_iv]

# 95%信頼区間
plt.bar(labels, coefs, yerr=1.96*np.array(errors), 
        capsize=10, color=['#3498db', '#e74c3c'], alpha=0.7)
plt.axhline(y=0, linestyle='--', color='gray', linewidth=1)
plt.ylabel('Coefficient on Muslim Population Share', fontsize=12)
plt.title('Figure 2: OLS vs IV Coefficient Comparison', fontsize=14)
plt.grid(axis='y', alpha=0.3)
plt.savefig('outputs/figures/figure2_coef_comparison.png', dpi=300, bbox_inches='tight')
print("✅ Figure 2: outputs/figures/figure2_coef_comparison.png")

# ================================
# Figure 3: 生の散布図
# ================================
plt.figure(figsize=(8, 6))
sc = plt.scatter(df['Muslims'], df['log_co2'], 
                c=df['log_gdp'], cmap='viridis', alpha=0.7, s=60)
plt.colorbar(sc, label='log(GDP)')
plt.xlabel('Muslim Population (%)', fontsize=12)
plt.ylabel('log(CO₂ per capita)', fontsize=12)
plt.title('Figure 3: Muslim Share vs CO₂ Emissions', fontsize=14)
plt.grid(alpha=0.3)
plt.savefig('outputs/figures/figure3_scatter.png', dpi=300, bbox_inches='tight')
print("✅ Figure 3: outputs/figures/figure3_scatter.png")

print("\n" + "=" * 60)
print("✅ すべてのFigureを作成完了しました！")
print("保存場所: outputs/figures/")
print("  - figure1_first_stage.png")
print("  - figure2_coef_comparison.png")
print("  - figure3_scatter.png")
print("=" * 60)