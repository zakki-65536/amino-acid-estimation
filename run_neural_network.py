import subprocess
import re
from datetime import datetime
import requests
import os
from dotenv import load_dotenv
import json

def execute_python_file(file_path_py, num_executions, epochs, file_path_excel, train_data_ratio):
    accuracy_list = []
    loss_list = []
    all_outputs = []

    for i in range(num_executions):
        try:
            result = subprocess.run(
                ["python", file_path_py, str(epochs), str(file_path_excel), str(train_data_ratio), str(result_path_excel), str(i+1).zfill(3)],
                capture_output=True,
                text=True,
                check=True
            )
            output = result.stdout.strip()
            all_outputs.append(f"Run {i+1}:\n{output}\n")

            # 各行を調べて正解率と損失を抽出
            isPrint=False
            for line in output.splitlines():
                acc_match = re.search(r"予測の正解率.*?([\d.]+)%", line)
                loss_match = re.search(r"学習の最終損失値 \(loss\): ([\d.]+)", line)
                print_start = re.search(r".*print.*start.*", line)
                print_end = re.search(r".*print.*end.*", line)

                if acc_match:
                    accuracy = float(acc_match.group(1))
                    accuracy_list.append(accuracy)

                if loss_match:
                    loss = float(loss_match.group(1))
                    loss_list.append(loss)

                if print_start:
                    isPrint=True
                elif print_end:
                    isPrint=False
                elif isPrint:
                    print(line)

        except subprocess.CalledProcessError as e:
            print(f"エラー: 実行{i+1}回目に失敗しました\n{e.stderr}")

    return accuracy_list, loss_list, all_outputs


# 実行するPythonファイルのパス
file_path_py = r"neural_network.py"
# 実行回数 この回数実行して平均をとる
num_executions=100
# 学習回数のリスト
# epochs_list=[1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
epochs_list=[100]

# データを格納しているExcelファイル
file_path_excel = 'data/data_13項目_空腹時_female_外れ値除去_287.xlsx'
# 全体のデータ数に対する学習用データ数の割合
train_data_ratio=0.8
# 出力結果を保存するExcelファイル
result_path_excel = 'data/result_13項目_空腹時_female_287_100回.xlsx'

# ファイル名と実行開始時刻を表示
print(f"file: {file_path_excel}")
response_str=f"file: {file_path_excel}\n"
dt=datetime.now()
datetime_str = dt.strftime("%m/%d %H:%M:%S")
print(f"{datetime_str} start")
response_str+=f"{datetime_str} start\n"

for epochs in epochs_list:
    # 10回実行して結果を取得
    accuracies, losses, outputs = execute_python_file(file_path_py, num_executions, epochs, file_path_excel, train_data_ratio)

    # 各実行の出力表示（任意）
    #for output in outputs:
    #    print(output)

    # 結果集計と表示
    if accuracies and losses:
        dt=datetime.now()
        datetime_str = dt.strftime("%m/%d %H:%M:%S")

        avg_accuracy = sum(accuracies) / len(accuracies)
        avg_loss = sum(losses) / len(losses)

        print(f"{datetime_str} {epochs} {avg_accuracy:.2f}% {avg_loss:.4f}")
        response_str+=f"{datetime_str} {epochs} {avg_accuracy:.2f}% {avg_loss:.4f}\n"
    else:
        print("正解率または損失値の取得に失敗しました。")

# メール通知APIを叩く
load_dotenv()
API_URL = os.getenv("API_URL")
API_URL_FILE = os.getenv("API_URL_FILE")
API_KEY = os.getenv("API_KEY")
HTTP_PROXY = os.getenv("HTTP_PROXY")
HTTPS_PROXY = os.getenv("HTTPS_PROXY")

if not API_URL or not API_KEY:
    print("API_URL, API_KEYが設定されていません")
else:
    header = {
        "Authorization": f"Bearer {API_KEY}",
        "Content-Type": "application/json"
    }
    payload = {
        'subject': "[amino-acid-estimation] 実行結果",
        'body': response_str
    }
    proxies = {
        "http": HTTP_PROXY,
        "https": HTTPS_PROXY,
    }
    
    with open(result_path_excel, "rb") as f:
        files = {
            # PHP側は最初のファイルフィールドを拾うので、name は "excel" でも "file" でもOK
            "excel": (os.path.basename(result_path_excel), f,
                "application/vnd.openxmlformats-officedocument.spreadsheetml.sheet")
        }

        try:
            # プロキシ環境
            res = requests.post(API_URL, headers=header, json=payload, proxies=proxies)
            res_json = res.json()
            print(res_json["message"])
            res = requests.post(API_URL_FILE, files=files, proxies=proxies, timeout=60)
            res_json = res.json()
            print(res_json["message"])
        except requests.exceptions.RequestException:
            try:
                # プロキシ環境でない
                res = requests.post(API_URL, headers=header, json=payload, timeout=10)
                res_json = res.json()
                print(res_json["message"])
                res = requests.post(API_URL_FILE, files=files, timeout=60)
                res_json = res.json()
                print(res_json["message"])
            except requests.exceptions.RequestException as e:
                # エラー
                print("Error:", e)