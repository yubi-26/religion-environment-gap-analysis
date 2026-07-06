import pandas as pd
import numpy as np
import requests
import time
import os

print("=== 🔄 世界銀行API・女性労働力率の再回収 ===")

# 既存のデータを読み込み
merged_df = pd.read_csv('data/processed/merged_2020.csv')
key_col = 'iso_code' if 'iso_code' in merged_df.columns else 'Countrycode'

# すでにカラムが存在する場合は一旦ドロップして綺麗にする
if 'female_labor_ratio' in merged_df.columns:
    merged_df = merged_df.drop(columns=['female_labor_ratio'])

# サーバーを驚かせないように少し待つ
print("サーバーの負荷を避けるため、1秒待機してからリクエストします...")
time.sleep(1)

ind_code = 'SL.TLF.CACT.FE.ZS'
url = f"https://api.worldbank.org/v2/country/all/indicator/{ind_code}?date=2020&format=json&per_page=300"

try:
    response = requests.get(url)
    if response.status_code == 200:
        json_data = response.json()
        records = json_data[1]
        
        parsed_data = []
        for record in records:
            iso3 = record['countryiso3code']
            val = record['value']
            if iso3 and val is not None:
                parsed_data.append({key_col: iso3, 'female_labor_ratio': val})
        
        df_labor = pd.DataFrame(parsed_data)
        
        # メインデータにマージ
        final_df = pd.merge(merged_df, df_labor, on=key_col, how='left')
        
        # 重複カラムを排除して保存
        final_df = final_df.loc[:, ~final_df.columns.duplicated()]
        final_df.to_csv('data/processed/merged_2020.csv', index=False)
        
        print("\n=============================================")
        print(f"🎉 リベンジ成功！現在の総カラム数: {len(final_df.columns)}")
        print("揃ったジェンダー指標一覧:")
        print([col for col in final_df.columns if 'female' in col])
        print("=============================================\n")
    else:
        print(f"❌ APIエラー (ステータスコード: {response.status_code})")
except Exception as e:
    print(f"❌ 再度エラーが発生しました: {e}")
    print("通信状況が不安定なため、このまま進めるか少し時間を空けてください。")
