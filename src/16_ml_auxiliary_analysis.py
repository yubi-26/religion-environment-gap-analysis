"""
Task 2: 限定的ML補助分析（確認版）
"""

import pandas as pd
import numpy as np
from sklearn.linear_model import RidgeCV
from sklearn.ensemble import RandomForestRegressor
from sklearn.model_selection import cross_val_score, KFold
from sklearn.inspection import permutation_importance
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("Task 2: 限定的ML補助分析（確認版）")
print("=" * 70)

# ========================================
# 1. データ読み込みと確認
# ========================================
df = pd.read_csv('data/processed/merged_epi_full_2020.csv')

print("=== データ確認 ===")
print(f"データ形状: {df.shape}")
print("Muslims列の要約:")
print(df['Muslims'].describe())

print("\n=== 先頭5カ国 ===")
print(df[['country', 'Muslims', 'gdp', 'population']].head())

# ========================================
# 2. 変数作成
# ========================================
if 'gdp_per_capita' in df.columns:
    df['log_gdp_per_capita'] = np.log(df['gdp_per_capita'])
else:
    df['log_gdp_per_capita'] = np.log(df['gdp'] / df['population'])

df['log_population'] = np.log(df['population'] + 1)
df['log_co2_per_capita'] = np.log1p(df['co2_per_capita'])
df['energy_per_capita_scaled'] = df['energy_per_capita'] / 1000

# oil_exporter（簡易版）
oil_countries = ['Saudi Arabia', 'Qatar', 'United Arab Emirates', 'Kuwait', 
                 'Brunei', 'Oman', 'Bahrain', 'Libya', 'Algeria', 'Nigeria']
df['oil_exporter'] = df['country'].isin(oil_countries).astype(int)

# 特徴量
features = ['Muslims', 'log_gdp_per_capita', 'log_population',
            'energy_per_capita_scaled', 'oil_exporter']

X = df[features].dropna()
y = df.loc[X.index, 'log_co2_per_capita']

print(f"\nサンプル数: {len(X)}")
print(f"特徴量数: {len(features)}")
print(f"特徴量: {features}")

# ========================================
# 3. 交差検証設定
# ========================================
kf = KFold(n_splits=5, shuffle=True, random_state=42)

# ========================================
# 4. Ridge回帰
# ========================================
ridge = RidgeCV(alphas=[0.1, 1, 10, 100])
ridge_scores = cross_val_score(ridge, X, y, cv=kf, scoring='r2')

print("\n=== Ridge回帰 ===")
print(f"CV R² (平均): {ridge_scores.mean():.4f}")
print(f"CV R² (標準偏差): {ridge_scores.std():.4f}")
print(f"CV R² (範囲): {ridge_scores.min():.4f} 〜 {ridge_scores.max():.4f}")

# ========================================
# 5. Random Forest
# ========================================
rf = RandomForestRegressor(
    n_estimators=100,
    max_depth=4,
    min_samples_leaf=5,
    random_state=42
)
rf_scores = cross_val_score(rf, X, y, cv=kf, scoring='r2')

print("\n=== Random Forest ===")
print(f"CV R² (平均): {rf_scores.mean():.4f}")
print(f"CV R² (標準偏差): {rf_scores.std():.4f}")

# ========================================
# 6. Feature Ablation
# ========================================
X_no_religion = X.drop(columns=['Muslims'])
rf_no_religion = RandomForestRegressor(
    n_estimators=100,
    max_depth=4,
    min_samples_leaf=5,
    random_state=42
)
rf_no_religion_scores = cross_val_score(rf_no_religion, X_no_religion, y, cv=kf, scoring='r2')

print("\n=== Feature Ablation ===")
print(f"RF (全特徴量) CV R²: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}")
print(f"RF (Muslims除外) CV R²: {rf_no_religion_scores.mean():.4f} ± {rf_no_religion_scores.std():.4f}")

gain = rf_scores.mean() - rf_no_religion_scores.mean()
print(f"\nMuslims追加によるR²増加: {gain:.4f}")

if abs(gain) < rf_scores.std():
    print("\n結論: 宗教変数の追加効果は標準偏差の範囲内。")
    print("       統計的に意味のある改善とは言えない。")
else:
    print("\n結論: 宗教変数の追加効果は標準偏差を超える。")
    print("       何らかの貢献がある可能性。")

# ========================================
# 7. 変数重要度（標準）
# ========================================
rf_full = RandomForestRegressor(
    n_estimators=100,
    max_depth=4,
    min_samples_leaf=5,
    random_state=42
)
rf_full.fit(X, y)

importance = pd.DataFrame({
    'feature': features,
    'importance': rf_full.feature_importances_
}).sort_values('importance', ascending=False)

print("\n=== 変数重要度（標準） ===")
print(importance.to_string(index=False))

# ========================================
# 8. Permutation Importance（追加）
# ========================================
print("\n=== Permutation Importance（30回繰り返し） ===")
result = permutation_importance(
    rf_full, X, y,
    n_repeats=30,
    random_state=42
)

importance_perm = pd.DataFrame({
    'feature': features,
    'importance_mean': result.importances_mean,
    'importance_std': result.importances_std
}).sort_values('importance_mean', ascending=False)

print(importance_perm.to_string(index=False))

# ========================================
# 9. 結果保存
# ========================================
with open('outputs/tables/ml_auxiliary_results.txt', 'w', encoding='utf-8') as f:
    f.write("Task 2: 限定的ML補助分析結果\n")
    f.write("=" * 70 + "\n\n")
    f.write(f"サンプル数: {len(X)}\n")
    f.write(f"特徴量数: {len(features)}\n")
    f.write(f"特徴量: {features}\n\n")
    
    f.write("=== Ridge回帰 ===\n")
    f.write(f"CV R² (平均): {ridge_scores.mean():.4f}\n")
    f.write(f"CV R² (標準偏差): {ridge_scores.std():.4f}\n\n")
    
    f.write("=== Random Forest ===\n")
    f.write(f"CV R² (平均): {rf_scores.mean():.4f}\n")
    f.write(f"CV R² (標準偏差): {rf_scores.std():.4f}\n\n")
    
    f.write("=== Feature Ablation ===\n")
    f.write(f"RF (全特徴量) CV R²: {rf_scores.mean():.4f} ± {rf_scores.std():.4f}\n")
    f.write(f"RF (Muslims除外) CV R²: {rf_no_religion_scores.mean():.4f} ± {rf_no_religion_scores.std():.4f}\n")
    f.write(f"Muslims追加によるR²増加: {gain:.4f}\n\n")
    
    if abs(gain) < rf_scores.std():
        f.write("結論: 宗教変数の追加効果は標準偏差の範囲内。統計的に意味のある改善とは言えない。\n\n")
    else:
        f.write("結論: 宗教変数の追加効果は標準偏差を超える。何らかの貢献がある可能性。\n\n")
    
    f.write("=== 変数重要度（標準） ===\n")
    f.write(importance.to_string(index=False) + "\n\n")
    
    f.write("=== Permutation Importance ===\n")
    f.write(importance_perm.to_string(index=False))

print("\n✅ 保存: outputs/tables/ml_auxiliary_results.txt")
print("=" * 70)