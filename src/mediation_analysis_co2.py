"""
環境経路分析（Environmental Pathway Analysis）
従属変数: co2_per_capita（EPIとは独立）
Bonferroni補正付き
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("環境経路分析（Environmental Pathway Analysis）")
print("従属変数: CO2 per capita（EPIとは独立）")
print("=" * 70)

# ========================================
# 1. データ読み込み
# ========================================
df = pd.read_csv('data/processed/merged_epi_full_2020.csv')
print(f"\nデータ形状: {df.shape}")

# ========================================
# 2. 変数定義
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
# 3. データ準備
# ========================================
df['log_gdp'] = np.log(df['gdp'])
df['log_co2'] = np.log1p(df['co2_per_capita'])

analysis_vars = [X_var, 'log_co2'] + list(mediators.keys()) + control_vars
df_clean = df[analysis_vars].dropna()
print(f"\n欠損除去後のサンプル数: {len(df_clean)}")

if len(df_clean) < 30:
    print("⚠️ サンプル数が少なすぎます。")
    exit()

# ========================================
# 4. 標準化
# ========================================
scale_vars = [X_var, 'log_co2'] + list(mediators.keys()) + control_vars
scaler = StandardScaler()
df_clean[scale_vars] = scaler.fit_transform(df_clean[scale_vars])
print("✅ 標準化完了（平均=0, 標準偏差=1）")

# ========================================
# 5. 経路分析
# ========================================
results = []

for mediator, label in mediators.items():
    print(f"\n{'='*70}")
    print(f"経路: {label} ({mediator})")
    print('='*70)
    
    cols = [X_var, mediator, 'log_co2'] + control_vars
    analysis_df = df_clean[cols].dropna()
    n = len(analysis_df)
    print(f"有効サンプル数: {n}")
    
    if n < 30:
        print("⚠️ サンプル数不足。スキップ。")
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
        'indirect_effect': indirect,
        'total_effect': total,
    })
    
    print(f"\n【{label}】")
    print(f"  Path A (Muslims → {mediator}): {model_A.params[X_var]:.4f} (p={model_A.pvalues[X_var]:.4f})")
    print(f"  Path B ({mediator} → log_CO2): {model_B.params[mediator]:.4f} (p={model_B.pvalues[mediator]:.4f})")
    print(f"  Path C' (Muslims → log_CO2): {model_C.params[X_var]:.4f} (p={model_C.pvalues[X_var]:.4f})")
    print(f"  間接効果: {indirect:.4f}")

# ========================================
# 6. Bonferroni補正 + 保存
# ========================================
results_df = pd.DataFrame(results)

num_tests = len(results_df)

results_df['path_a_p_bonferroni'] = (
    results_df['path_a_p'] * num_tests
).clip(upper=1.0)

results_df['path_b_p_bonferroni'] = (
    results_df['path_c_mediator_p'] * num_tests
).clip(upper=1.0)

results_df['robust_mediation'] = (
    (results_df['path_a_p_bonferroni'] < 0.05) &
    (results_df['path_b_p_bonferroni'] < 0.05)
)

results_df.to_csv(
    'outputs/tables/environmental_pathway_analysis_results.csv',
    index=False
)

print("\n" + "=" * 70)
print(f"Bonferroni補正（α = 0.05 / {num_tests} = {0.05/num_tests:.4f}）")
print("=" * 70)

for _, row in results_df.iterrows():
    status = "✅ Robust" if row['robust_mediation'] else "❌ Not robust"
    print(f"{row['label']}:")
    print(f"  Path A corrected: p={row['path_a_p_bonferroni']:.4f}")
    print(f"  Path B corrected: p={row['path_b_p_bonferroni']:.4f}")
    print(f"  → {status}")

robust = results_df[results_df['robust_mediation']]
if len(robust) > 0:
    print("\n✅ Bonferroni補正後も頑健な経路:")
    for _, row in robust.iterrows():
        print(f"  - {row['label']}: 間接効果 = {row['indirect_effect']:.4f}")
else:
    print("\n⚠️ Bonferroni補正後、頑健な経路は見つかりませんでした。")

print("\n✅ 保存: outputs/tables/environmental_pathway_analysis_results.csv")
print("=" * 70)