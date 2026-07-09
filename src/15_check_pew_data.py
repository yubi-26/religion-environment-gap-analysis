"""
Pewデータ構造確認（修正版）
"""

import pandas as pd

# 正しいファイルパスで読み込み（CSV形式）
pew_raw = pd.read_csv('data/raw/pew_religion_data.csv')

print("=== Pewデータ列名 ===")
print(pew_raw.columns.tolist())

print("\n=== Pewデータ先頭5行 ===")
print(pew_raw.head())

print("\n=== Pewデータ情報 ===")
print(pew_raw.info())

print("\n=== ユニークな年 ===")
if 'year' in pew_raw.columns:
    print(pew_raw['year'].unique())
else:
    print("'year'列が見つかりません。代わりに:")
    for col in pew_raw.columns:
        if 'year' in col.lower():
            print(f"  {col}: {pew_raw[col].unique()}")

print("\n=== ユニークなレベル ===")
if 'level' in pew_raw.columns:
    print(pew_raw['level'].unique())
else:
    print("'level'列が見つかりません。")