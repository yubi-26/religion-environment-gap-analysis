"""
媒介分析（Mediation Analysis）- 完全版
標準化 + 複数Mediator
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
from sklearn.preprocessing import StandardScaler

print("=" * 60)
print("媒介分析開始（Mediation Analysis）- 完全版")
print("=" * 60)

# ========================================
# 1. データ読み込み
# ========================================
df = pd.read_csv('data/processed/merged_epi_full_2020.csv')
print(f"データ形状: {df.shape}")

# ========================================
# 2. 変数定義
# ========================================
X_var = 'Muslims'
Y_var = 'epi_score'
control_vars = ['log_gdp']

# 媒介変数（メカニズム分析）
mediators = {
    'red_list_index': 'Red List Index (絶滅危惧種指数)',
    'species_habitat': 'Species Habitat Index (種生息地指数)',
    'species_protection': 'Species Protection Index (種保護指数)',
    'pesticide_risk': 'Pesticide Risk (殺虫剤リスク)',
    'biodiversity_habitat': 'Biodiversity Habitat (生物多様性・生息地)',
    'air_quality': 'Air Quality (大気質)',
}

print(f"独立変数: {X_var}")
print(f"従属変数: {Y_var}")
print(f"媒介変数: {list(mediators.keys())}")

# ========================================
# 3. log_gdpを作成
# ========================================
df['log_gdp'] = np.log(df['gdp'])

# ========================================
# 4. 標準化（係数比較のため）
# ========================================
scale_vars = [X_var, Y_var] + list(mediators.keys()) + control_vars
scaler = StandardScaler()
df[scale_vars] = scaler.fit_transform(df[scale_vars])

print(f"\n標準化完了（平均=0, 標準偏差=1）")

# ========================================
# 5. 媒介分析
# ========================================
results = []

for mediator, label in mediators.items():
    print(f"\n{'='*60}")
    print(f"媒介変数: {label} ({mediator})")
    print('='*60)
    
    analysis_cols = [X_var, mediator, Y_var] + control_vars
    analysis_df = df[analysis_cols].dropna()
    print(f"有効サンプル数: {len(analysis_df)}")
    
    if len(analysis_df) < 30:
        print("⚠️ サンプル数が少なすぎます。スキップします。")
        continue
    
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
    
    # 結果を保存
    results.append({
        'mediator': mediator,
        'label': label,
        'n': len(analysis_df),
        'path_a_coef': model_A.params[X_var],
        'path_a_p': model_A.pvalues[X_var],
        'path_b_coef': model_B.params[mediator],
        'path_b_p': model_B.pvalues[mediator],
        'path_c_coef': model_C.params[X_var],
        'path_c_p': model_C.pvalues[X_var],
        'path_c_mediator_coef': model_C.params[mediator],
        'path_c_mediator_p': model_C.pvalues[mediator],
    })
    
    print(f"\n[結果] {label}")
    print(f"  Path A (Muslims → {mediator}): {model_A.params[X_var]:.4f} (p={model_A.pvalues[X_var]:.4f})")
    print(f"  Path B ({mediator} → EPI): {model_B.params[mediator]:.4f} (p={model_B.pvalues[mediator]:.4f})")
    print(f"  Path C' (Muslims → EPI, 制御後): {model_C.params[X_var]:.4f} (p={model_C.pvalues[X_var]:.4f})")
    
    # 間接効果を計算
    indirect = model_A.params[X_var] * model_C.params[mediator]
    print(f"  間接効果 (Indirect): {indirect:.4f}")

# ========================================
# 6. 結果保存
# ========================================
results_df = pd.DataFrame(results)
results_df.to_csv('outputs/tables/mediation_results_final.csv', index=False)
print("\n✅ 保存: outputs/tables/mediation_results_final.csv")
print("=" * 60)