import pandas as pd
from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans
import numpy as np

file_path_excel = 'data/data_クラスタリング用_13項目_空腹時_female_319.xlsx'
file_path_result = 'data/result_クラスタリング_13項目_空腹時_female_319.xlsx'

# データの読み込み
data = pd.read_excel(file_path_excel)
data.columns = data.columns.astype(str)  # カラム名を文字列に変換

# 標準化
scaler=StandardScaler() 
data_scaled=scaler.fit_transform(data) 

# クラスタリング
km = KMeans(n_clusters=2,            # クラスターの個数         # セントロイドの初期値をランダムに設定  
            n_init=10,               # 異なるセントロイドの初期値を用いたk-meansの実行回数
            max_iter=300,            # k-meansアルゴリズムの内部の最大イテレーション回数  
            tol=1e-04,               # 収束と判定するための相対的な許容誤差 
            random_state=0)          # セントロイドの初期化に用いる乱数発生器の状態
cluster = km.fit_predict(data)

# print(list(cluster))
data['クラス']=list(cluster)
print(data)
pd.DataFrame(data).to_excel(file_path_result,index=False)