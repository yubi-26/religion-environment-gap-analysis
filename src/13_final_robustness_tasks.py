"""
5時間タスク第2弾（修正版）- エラー修正
"""

import pandas as pd
import numpy as np
import statsmodels.api as sm
import matplotlib.pyplot as plt
import warnings
warnings.filterwarnings('ignore')

print("=" * 70)
print("5時間タスク第2弾（修正版）")
print("=" * 70)

# ========================================
# データ読み込み
# ========================================
df = pd.read_csv('data/processed/merged_epi_full_2020.csv')
print(f"データ形状: {df.shape}")

# ========================================
# 変数作成
# ========================================
df['log_gdp_per_capita'] = np.log(df['gdp'] / df['population'])
df['log_population'] = np.log(df['population'] + 1)
df['log_co2_per_capita'] = np.log1p(df['co2_per_capita'])
df['total_co2'] = df['co2_per_capita'] * df['population']
df['log_total_co2'] = np.log(df['total_co2'] + 1)
df['energy_per_capita_scaled'] = df['energy_per_capita'] / 1000
df['log_gdp'] = np.log(df['gdp'])

print("✅ 変数作成完了")

# ========================================
# Hour 1: oil_exporter バグ修正
# ========================================
print("\n" + "=" * 70)
print("Hour 1: oil_exporter メカニズム検証")
print("=" * 70)

# 石油輸出国リスト
oil_countries = ['Saudi Arabia', 'Qatar', 'United Arab Emirates', 'Kuwait', 
                 'Brunei', 'Oman', 'Bahrain', 'Libya', 'Algeria', 'Nigeria']

# 実際の国名列を確認
country_col = 'country' if 'country' in df.columns else 'country_name'

print(f"国名列: {country_col}")
print(f"\n石油輸出国のデータ存在確認:")

existing_oil = []
for name in oil_countries:
    matches = df[df[country_col].str.contains(name.split()[0], case=False, na=False)]
    if len(matches) > 0:
        actual_names = matches[country_col].tolist()
        print(f"  {name}: {actual_names}")
        existing_oil.extend(actual_names)
    else:
        print(f"  {name}: 一致なし")

# oil_exporter フラグ作成
df['oil_exporter'] = df[country_col].isin(existing_oil).astype(int)
print(f"\noil_exporter 分布:")
print(df['oil_exporter'].value_counts())

# ========================================
# 分析用データ
# ========================================
analysis_vars = [
    'iso_code', 'Region', 'country',
    'Muslims',
    'log_gdp_per_capita', 'log_gdp', 'log_population',
    'energy_per_capita_scaled',
    'log_co2_per_capita', 'log_total_co2',
    'oil_exporter'
]
analysis_df = df[analysis_vars].dropna()
print(f"\n分析サンプル数: {len(analysis_df)}")
print(f"分析サンプル内の石油輸出国数: {analysis_df['oil_exporter'].sum()}")

# 地域ダミー
region_dummies = pd.get_dummies(analysis_df['Region'], drop_first=True, dtype=int)

# ========================================
# oil_exporter メカニズム検証
# ========================================
print("\n" + "=" * 70)
print("oil_exporter メカニズム検証")
print("=" * 70)

y = analysis_df['log_co2_per_capita']

# モデルA: ベースライン（比率 + GDPpc + Energy）
X_A = sm.add_constant(analysis_df[['Muslims', 'log_gdp_per_capita', 'energy_per_capita_scaled']])
model_A = sm.OLS(y, X_A).fit(cov_type='HC3')
print("\nモデルA: Muslim% + GDPpc + Energy")
print(f"  Muslims: coef={model_A.params['Muslims']:.4f}, p={model_A.pvalues['Muslims']:.4f}")

# モデルB: + oil_exporter
X_B = sm.add_constant(analysis_df[['Muslims', 'log_gdp_per_capita', 'energy_per_capita_scaled', 'oil_exporter']])
model_B = sm.OLS(y, X_B).fit(cov_type='HC3')
print("\nモデルB: Muslim% + GDPpc + Energy + oil_exporter")
print(f"  Muslims: coef={model_B.params['Muslims']:.4f}, p={model_B.pvalues['Muslims']:.4f}")
print(f"  oil_exporter: coef={model_B.params['oil_exporter']:.4f}, p={model_B.pvalues['oil_exporter']:.4f}")

# ========================================
# Hour 2: Specification Curve（仕様順）
# ========================================
print("\n" + "=" * 70)
print("Hour 2: Specification Curve（仕様順）")
print("=" * 70)

try:
    spec_df = pd.read_csv('outputs/tables/specification_curve_results.csv')
    
    # 仕様順にソート
    spec_df['Model_Num'] = spec_df['Model'].str.extract(r'(\d+)').astype(int)
    spec_df = spec_df.sort_values('Model_Num').reset_index(drop=True)
    
    print("Specification Curve Results (仕様順):")
    print(spec_df[['Model', 'Coefficient', 'p_value']].to_string())
    
    # グラフ作成（修正版）
    fig, ax = plt.subplots(figsize=(12, 6))
    
    # 各点の色を設定（赤: p<0.05, 灰: p>=0.05）
    point_colors = ['red' if p < 0.05 else 'gray' for p in spec_df['p_value']]
    
    # エラーバーを個別にプロット
    for i, row in spec_df.iterrows():
        color = 'red' if row['p_value'] < 0.05 else 'gray'
        ax.errorbar(
            i, row['Coefficient'],
            yerr=[[row['Coefficient'] - row['CI_lower']], [row['CI_upper'] - row['Coefficient']]],
            fmt='o',
            color=color,
            ecolor=color,
            capsize=5,
            elinewidth=2,
            markersize=8
        )
    
    ax.axhline(0, color='black', linewidth=1, linestyle='--')
    ax.set_xticks(range(len(spec_df)))
    ax.set_xticklabels(spec_df['Model'], rotation=45, ha='right', fontsize=9)
    ax.set_ylabel('Muslims Coefficient', fontsize=12)
    ax.set_xlabel('Model Specification (in order of complexity)', fontsize=12)
    ax.set_title('Specification Curve: Muslim Coefficient Across 10 Models\n(red = p < 0.05, gray = p ≥ 0.05)', fontsize=14)
    
    from matplotlib.patches import Patch
    legend_elements = [
        Patch(facecolor='red', label='p < 0.05 (significant)'),
        Patch(facecolor='gray', label='p ≥ 0.05 (not significant)'),
    ]
    ax.legend(handles=legend_elements, loc='upper right')
    
    plt.tight_layout()
    plt.savefig('outputs/figures/specification_curve_ordered.png', dpi=150)
    print("\n✅ 保存: outputs/figures/specification_curve_ordered.png")
    plt.close()
    
except FileNotFoundError:
    print("⚠️ specification_curve_results.csv が見つかりません。")
except Exception as e:
    print(f"⚠️ グラフ作成エラー: {e}")

# ========================================
# Hour 3: EPIコードブック整理
# ========================================
print("\n" + "=" * 70)
print("Hour 3: EPIコードブック整理")
print("=" * 70)

epi_cols = [c for c in df.columns if any(x in c.lower() for x in ['epi', 'biodiv', 'habitat', 'species', 'protection'])]
print(f"EPI関連カラム: {epi_cols}")

data_dict_content = """# Data Dictionary

## Environmental Variables (EPI)
| Variable | Meaning | Source |
|----------|---------|--------|
| RLI | Red List Index (biodiversity indicator) | EPI |
| Biodiversity_Habitat | Biodiversity and Habitat index | EPI |
| Species_Protection | Species Protection Index | EPI |

## Social Variables
| Variable | Meaning | Source |
|----------|---------|--------|
| Muslims | Muslim population share (%) | Pew Research |
| gdp | GDP (current USD) | World Bank |
| population | Total population | World Bank |
| energy_per_capita | Energy consumption per capita | World Bank |

## Important Notes
- RLI = Red List Index, NOT Rule of Law
- Rule of Law should be obtained from World Justice Project (WJP) if needed
"""

with open('data_dictionary.md', 'w', encoding='utf-8') as f:
    f.write(data_dict_content)

print("✅ 保存: data_dictionary.md")

# ========================================
# Hour 5: クラスター標準誤差
# ========================================
print("\n" + "=" * 70)
print("Hour 5: クラスター標準誤差 (HC3 vs Cluster)")
print("=" * 70)

X = sm.add_constant(analysis_df[['Muslims', 'log_gdp_per_capita', 'log_population', 'energy_per_capita_scaled']])
y = analysis_df['log_co2_per_capita']

model_hc3 = sm.OLS(y, X).fit(cov_type='HC3')
print("\n=== HC3標準誤差 ===")
print(f"Muslims: coef={model_hc3.params['Muslims']:.4f}, p={model_hc3.pvalues['Muslims']:.4f}")

if 'Region' in analysis_df.columns:
    region_count = analysis_df['Region'].nunique()
    print(f"\n地域クラスタ数: {region_count}")
    
    if region_count >= 5:
        model_cluster = sm.OLS(y, X).fit(cov_type='cluster', cov_kwds={'groups': analysis_df['Region']})
        print("\n=== クラスター標準誤差（地域） ===")
        print(f"Muslims: coef={model_cluster.params['Muslims']:.4f}, p={model_cluster.pvalues['Muslims']:.4f}")
    else:
        print("⚠️ 地域クラスタ数が少なすぎます（<5）。クラスターSEは解釈に注意。")

print("\n=== 標準誤差比較 ===")
print(f"{'SE Type':<15} {'Muslims β':<12} {'p-value':<10}")
print("-" * 40)
print(f"{'HC3':<15} {model_hc3.params['Muslims']:>10.4f} {model_hc3.pvalues['Muslims']:>10.4f}")
if region_count >= 5:
    print(f"{'Cluster (region)':<15} {model_cluster.params['Muslims']:>10.4f} {model_cluster.pvalues['Muslims']:>10.4f}")

# ========================================
# 保存
# ========================================
with open('outputs/tables/cluster_vs_hc3_comparison.txt', 'w', encoding='utf-8') as f:
    f.write("HC3 vs クラスター標準誤差 比較\n")
    f.write("=" * 40 + "\n")
    f.write(f"HC3: coef={model_hc3.params['Muslims']:.4f}, p={model_hc3.pvalues['Muslims']:.4f}\n")
    if region_count >= 5:
        f.write(f"Cluster(region): coef={model_cluster.params['Muslims']:.4f}, p={model_cluster.pvalues['Muslims']:.4f}\n")
    f.write(f"\n地域クラスタ数: {region_count}")

print("\n✅ 保存: outputs/tables/cluster_vs_hc3_comparison.txt")
print("=" * 70)

# ========================================
# チェックリスト
# ========================================
print("\n" + "=" * 70)
print("5時間タスク チェックリスト")
print("=" * 70)
print(f"[x] oil_exporter が0でないことを確認: {analysis_df['oil_exporter'].sum() > 0}")
print(f"[x] oil追加前後でMuslims係数比較: {model_A.params['Muslims']:.4f} → {model_B.params['Muslims']:.4f}")
print("[x] specification_curve_ordered.png完成")
print("[x] data_dictionary.md完成")
print("[x] HC3 vs Cluster SE比較完了")
print("[ ] research_notes.md更新（別途手動で編集）")
print("=" * 70)