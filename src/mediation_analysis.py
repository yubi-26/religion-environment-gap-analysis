import pandas as pd
import numpy as np
import statsmodels.api as sm
import os

print("=============================================")
print(" 🧬 媒介分析 (Mediation Analysis) の実行開始")
print("=============================================\n")

# データの読み込み
df = pd.read_csv('data/processed/merged_2020.csv')

# 経済のコントロールとして log(GDP) を追加
df['log_gdp'] = np.log(df['gdp'])

# ターゲットとなるジェンダー指標
gender_indicators = {
    'female_parl_ratio': '女性議員比率',
    'female_labor_ratio': '女性労働力率'
}

for ind, label in gender_indicators.items():
    print(f"--- 【検証経路】ジェンダー指標: {label} ({ind}) ---")
    
    # 必要な変数（CO2, 宗教, GDP, 対象ジェンダー指標）が全て揃っている国に絞る
    analysis_df = df[['co2_per_capita', 'Muslims', 'log_gdp', ind]].dropna()
    print(f"有効分析国数: {len(analysis_df)} カ国")
    
    if len(analysis_df) < 30:
        print("データ数が少なすぎるため、この指標の分析をスキップします。\n")
        continue
        
    y_co2 = analysis_df['co2_per_capita']
    
    # ステップ1: Path A (宗教 -> ジェンダー指標)
    # ※もし宗教がジェンダー不平等を強めるなら、女性比率は「下がる」はず（マイナスの係数を期待）
    X_pathA = sm.add_constant(analysis_df[['Muslims', 'log_gdp']])
    y_gender = analysis_df[ind]
    model_A = sm.OLS(y_gender, X_pathA).fit()
    
    # ステップ2: Path B & C' (宗教 + ジェンダー + GDP -> CO2)
    X_pathBC = sm.add_constant(analysis_df[['Muslims', ind, 'log_gdp']])
    model_BC = sm.OLS(y_co2, X_pathBC).fit()
    
    # 結果のダイジェスト表示
    print("\n[結果ダイジェスト]")
    print(f" 1. Path A (Muslims -> {ind}):")
    print(f"    - 係数: {model_A.params['Muslims']:.4f} (p値: {model_A.pvalues['Muslims']:.4f})")
    
    print(f" 2. Path B ({ind} -> CO2):")
    print(f"    - 係数: {model_BC.params[ind]:.4f} (p値: {model_BC.pvalues[ind]:.4f})")
    
    print(f" 3. Direct Path C' (Muslims -> CO2 ※ジェンダーを考慮後):")
    print(f"    - 係数: {model_BC.params['Muslims']:.4f} (p値: {model_BC.pvalues['Muslims']:.4f})")
    print("-" * 50 + "\n")

print("🎉 媒介分析の計算が完了しました！")
