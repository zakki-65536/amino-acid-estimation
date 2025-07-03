# ニューラルネットワーク
# インストールは"pip install pandas tensorflow scikit-learn openpyxl numpy"

import pandas as pd
import tensorflow as tf
from sklearn.preprocessing import StandardScaler
from sklearn.model_selection import train_test_split
import numpy as np
from sklearn.metrics import mean_absolute_error, mean_squared_error
import sys

# コマンドライン引数
# エポック数 デフォルトは10
ep = int(sys.argv[1]) if len(sys.argv) > 1 else 10
# 実行するPythonファイルのパス
file_path_excel = sys.argv[2] if len(sys.argv) > 1 else ""
# 全体のデータ数に対する学習用データ数の割合 デフォルトは0.8
train_data_ratio = float(sys.argv[3]) if len(sys.argv) > 1 else 0.8

# Excelファイルからデータを読み込む
data = pd.read_excel(file_path_excel)

# Excelファイルの最終行と最終列を取得
num_data=data.shape[0]
num_feature=data.shape[1]-1

train_data_size=int(num_data*train_data_ratio)

# print("print epochs=", ep, ", file_path_excel=", file_path_excel, ", num_data=", num_data, ", train_data_size=", train_data_size)

# 特徴量と目的変数の設定
X = data.iloc[:, 0:num_feature]  # 特徴量 最終列以外
X.columns = X.columns.astype(str)  # カラム名を文字列に変換
y = data.iloc[:, num_feature]    # 目的変数 最終列

# 学習用データ8割とテスト用データ2割に分割
X_train, X_test, y_train, y_test = train_test_split(X, y, train_size=train_data_size) #train_size=学習用データの人数，random_state=42のように設定すると再現性がある

#print("X_train columns:", X_train.columns)
#print("X_test columns:", X_test.columns)

# 特徴量のスケーリング（標準化）
scaler_X = StandardScaler()
scaler_y = StandardScaler()

X_train_scaled = scaler_X.fit_transform(X_train)
X_test_scaled = scaler_X.transform(X_test)
y_train_scaled = scaler_y.fit_transform(y_train.values.reshape(-1, 1)).ravel()  # yを1次元に変換

# 学習データ（X_train）の平均と標準偏差
#print("X_trainの平均:\n", scaler_X.mean_)
#print("\nX_trainの標準偏差:\n", np.sqrt(scaler_X.var_))  # 標準偏差は分散の平方根

# テストデータ（X_test）の平均と標準偏差（参考用）
#print("\nX_testの平均:\n", X_test.mean().values)
#print("\nX_testの標準偏差:\n", X_test.std().values)

# 学習データ（y_train）の平均と標準偏差
#print("\ny_trainの平均:", scaler_y.mean_[0])
#print("y_trainの標準偏差:", np.sqrt(scaler_y.var_[0]))


# ニューラルネットワークモデルの構築
model = tf.keras.Sequential([
    tf.keras.layers.Dense(32, activation='relu', input_shape=(X_train_scaled.shape[1],)), #隠れ層1, ユニット数64 項目数の約2倍
    tf.keras.layers.Dense(8, activation='relu'), #隠れ層2, ユニット数32 隠れ層1の1/4
    #tf.keras.layers.Dense(15, activation='relu'), #隠れ層3, ユニット数16
    #tf.keras.layers.Dense(4, activation='relu'), #隠れ層4, ユニット数8
    tf.keras.layers.Dense(1)  # 出力層
])

# モデルのコンパイル
model.compile(optimizer='adam', loss=tf.keras.losses.Huber()) #損失関数loss='mse'

# モデルの学習
history = model.fit(X_train_scaled, y_train_scaled, epochs=ep, batch_size=128, verbose=0) #epochs=32,16,8 各10回max,min,ave

# 学習の最終loss値の取得
final_loss = history.history['loss'][-1]  # 最後のエポックの損失値

# 予測の実行
y_pred_scaled = model.predict(X_test_scaled).flatten()
y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()  # 予測結果を元のスケールに戻す

#y_pred_scaled = model.predict(X_train_scaled).flatten()
#y_pred = scaler_y.inverse_transform(y_pred_scaled.reshape(-1, 1)).ravel()  # 予測結果を元のスケールに戻す

# 絶対平均誤差と平均二乗誤差の計算
mae = mean_absolute_error(y_test, y_pred)
mse = mean_squared_error(y_test, y_pred)

#print(f"\n絶対平均誤差 (MAE): {mae:.2f}")
#print(f"平均二乗誤差 (MSE): {mse:.2f}")

# 予測値の表示
#print("\n予測結果:")
#for i in range(len(y_pred)):
    #print(f"予測値: {y_pred[i]:.2f}, 実際の値: {y_test.iloc[i]:.2f}")
    #print(f"予測値: {y_pred[i]:.2f}, 実際の値: {y_train.iloc[i]:.2f}")

# 正解率の計算 (+-5%以内を正解とする) 5%,10%,15%,5,10,15
tolerance = 0.05  # ±5%
correct_predictions = np.abs(y_pred - y_test.values) <= tolerance * y_test.values  
accuracy = np.mean(correct_predictions) * 100  # 正解率（%）

#correct_predictions = np.abs(y_pred - y_train) <= tolerance * y_train
#accuracy = np.mean(correct_predictions) * 100  # 正解率（%）

# 予測の正解率の表示
print(f"\n予測の正解率: {accuracy:.2f}%")

# 最終損失値の表示
print(f"学習の最終損失値 (loss): {final_loss:.4f}")
