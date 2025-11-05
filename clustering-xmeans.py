import pandas as pd
from sklearn.preprocessing import StandardScaler
from pyclustering.cluster.xmeans import xmeans, kmeans_plusplus_initializer
import numpy as np

# ファイルパス
file_path_excel = 'data/data_クラスタリング用_13項目_空腹時_female_319.xlsx'
file_path_result = 'data/result_xmeans_クラスタリング_13項目_空腹時_female.xlsx'

# データの読み込み
data = pd.read_excel(file_path_excel)
data.columns = data.columns.astype(str)  # カラム名を文字列に変換
print(data.head())

# 標準化
scaler = StandardScaler()
data_scaled = scaler.fit_transform(data)

# X-means クラスタリング
# 初期クラスタ中心（K-means++初期化）
initial_centers = kmeans_plusplus_initializer(data_scaled, 2).initialize()  # 初期クラスタ数=2（最小値）

# X-meansインスタンス作成
# kmax: 最大クラスタ数（自動的に2〜kmaxの範囲で最適なクラスタ数を探索）
xmeans_instance = xmeans(data_scaled, initial_centers, kmax=10)

# 学習（クラスタリング実行）
xmeans_instance.process()

# 結果の取得
clusters = xmeans_instance.get_clusters()   # 各クラスタのインデックスリスト
centers = xmeans_instance.get_centers()     # 各クラスタの重心

# クラスタラベルをデータに追加
labels = np.zeros(len(data_scaled))
for cluster_id, cluster_points in enumerate(clusters):
    labels[cluster_points] = cluster_id

data['クラス'] = labels.astype(int)
print(data.head())

# Excelに出力
data.to_excel(file_path_result, index=False)
print(f"結果を出力しました: {file_path_result}")