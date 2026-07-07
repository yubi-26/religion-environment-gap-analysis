"""
媒介分析（Mediation Analysis）- 構造的に正しい版
従属変数: co2_per_capita（EPIとは独立したアウトカム）
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("媒介分析（Mediation Analysis）- 構造的に正しい版")
print("従属変数: CO2 per capita（EPIとは独立）")
print("=" * 70)

# ========================================
# 1. データ読み込み
# ========================================
df = pd.read_csv('data/processed/merged_epi_full_2020.csv')
print(f"\nデータ形状: {df.shape}")

# ========================================
# 2. 利用可能変数の確認（デバッグ）
# ========================================
print("\n【利用可能変数一覧】")
print([c for c in df.columns if 'co2' in c.lower() or 'epi' in c.lower() or 'gdp' in c.lower()])

print("\n【CO2統計】")
print(df['co2_per_capita'].describe())

# ========================================
# 3. 変数定義
# ========================================
X_var = 'Muslims'
Y_var = 'co2_per_capita'
control_vars = ['log_gdp']

mediators = {
    'biodiversity_habitat': 'Biodiversity & Habitat',
    'air_quality': 'Air Quality',
    'red_list_index': 'Red List Index',
    'species_protection': 'Species Protection',
    'species_habitat': 'Species Habitat',
    'pesticide_risk': 'Pesticide Risk',
    'forests': 'Forests',
}

print(f"\n独立変数 (X): {X_var}")
print(f"従属変数 (Y): {Y_var}")
print(f"制御変数: {control_vars}")
print(f"媒介変数 (M): {list(mediators.keys())}")

# ========================================
# 4. データ準備（log変換）
# ========================================
df['log_gdp'] = np.log(df['gdp'])
df['log_co2'] = np.log1p(df['co2_per_capita'])  # log1pで安定化

print(f"\nlog_gdp 作成完了（欠損: {df['log_gdp'].isnull().sum()}件）")
print(f"log_co2 作成完了（欠損: {df['log_co2'].isnull().sum()}件）")

# ========================================
# 5. 分析用データセット
# ========================================
analysis_vars = [X_var, 'log_co2'] + list(mediators.keys()) + control_vars
df_clean = df[analysis_vars].dropna()
print(f"\n欠損除去後のサンプル数: {len(df_clean)}")

if len(df_clean) < 30:
    print("⚠️ サンプル数が少なすぎます。分析を終了します。")
    exit()

# ========================================
# 6. 標準化
# ========================================
scale_vars = [X_var, 'log_co2'] + list(mediators.keys()) + control_vars
scaler = StandardScaler()
df_clean[scale_vars] = scaler.fit_transform(df_clean[scale_vars])
print("✅ 標準化完了（平均=0, 標準偏差=1）")

# ========================================
# 7. 媒介分析
# ========================================
results = []

for mediator, label in mediators.items():
    print(f"\n{'='*70}")
    print(f"媒介変数: {label} ({mediator})")
    print('='*70)
    
    cols = [X_var, mediator, 'log_co2'] + control_vars
    analysis_df = df_clean[cols].dropna()
    n = len(analysis_df)
    print(f"有効サンプル数: {n}")
    
    if n < 30:
        print("⚠️ サンプル数が少なすぎます。スキップします。")
        continue
    
    # Path A: X → M
    X_A = sm.add_constant(analysis_df[[X_var] + control_vars])
    y_A = analysis_df[mediator]
    model_A = sm.OLS(y_A, X_A).fit(cov_type='HC3')
    
    # Path B: M → Y
    X_B = sm.add_constant(analysis_df[[mediator] + control_vars])
    y_B = analysis_df['log_co2']
    model_B = sm.OLS(y_B, X_B).fit(cov_type='HC3')
    
    # Path C': X + M → Y
    X_C = sm.add_constant(analysis_df[[X_var, mediator] + control_vars])
    y_C = analysis_df['log_co2']
    model_C = sm.OLS(y_C, X_C).fit(cov_type='HC3')
    
    # 間接効果
    indirect = model_A.params[X_var] * model_C.params[mediator]
    total = model_C.params[X_var] + indirect
    
    results.append({
        'mediator': mediator,
        'label': label,
        'n': n,
        'path_a_coef': model_A.params[X_var],
        'path_a_p': model_A.pvalues[X_var],
        'path_b_coef': model_B.params[mediator],
        'path_b_p': model_B.pvalues[mediator],
        'path_c_coef': model_C.params[X_var],
        'path_c_p': model_C.pvalues[X_var],
        'path_c_mediator_coef': model_C.params[mediator],
        'path_c_mediator_p': model_C.pvalues[mediator],
        'r_squared': model_C.rsquared,
        'indirect_effect': indirect,
        'total_effect': total,
    })
    
    print(f"\n【{label}】")
    print(f"  Path A (Muslims → {mediator}): {model_A.params[X_var]:.4f} (p={model_A.pvalues[X_var]:.4f})")
    print(f"  Path B ({mediator} → log_CO2): {model_B.params[mediator]:.4f} (p={model_B.pvalues[mediator]:.4f})")
    print(f"  Path C' (Muslims → log_CO2): {model_C.params[X_var]:.4f} (p={model_C.pvalues[X_var]:.4f})")
    print(f"  【間接効果】: {indirect:.4f}")
    print(f"  【総効果】: {total:.4f}")

# ========================================
# 8. 結果保存
# ========================================
results_df = pd.DataFrame(results)
results_df.to_csv('outputs/tables/mediation_co2_results.csv', index=False)
print(f"\n{'='*70}")
print("✅ 保存: outputs/tables/mediation_co2_results.csv")
print('='*70)

# ========================================
# 9. サマリー
# ========================================
print("\n" + "=" * 70)
print("結果サマリー")
print("=" * 70)

significant = results_df[
    (results_df['path_a_p'] < 0.05) & 
    (results_df['path_c_mediator_p'] < 0.05)
]

if len(significant) > 0:
    print("\n✅ 有意な媒介効果が見られた変数:")
    for _, row in significant.iterrows():
        print(f"  - {row['label']}: 間接効果 = {row['indirect_effect']:.4f}")
else:
    print("\n⚠️ 有意な媒介効果は見つかりませんでした。")

print("\n" + "=" * 70)