import subprocess
import re
from datetime import datetime

def execute_python_file(file_path_py, num_executions, epochs, file_path_excel, train_data_ratio):
    accuracy_list = []
    loss_list = []
    all_outputs = []

    for i in range(num_executions):
        try:
            result = subprocess.run(
                ["python", file_path_py, str(epochs), str(file_path_excel), str(train_data_ratio)],
                capture_output=True,
                text=True,
                check=True
            )
            output = result.stdout.strip()
            all_outputs.append(f"Run {i+1}:\n{output}\n")

            # 各行を調べて正解率と損失を抽出
            for line in output.splitlines():
                acc_match = re.search(r"予測の正解率.*?([\d.]+)%", line)
                loss_match = re.search(r"学習の最終損失値 \(loss\): ([\d.]+)", line)
                print_text = re.search(r".*print.*", line)

                if acc_match:
                    accuracy = float(acc_match.group(1))
                    accuracy_list.append(accuracy)

                if loss_match:
                    loss = float(loss_match.group(1))
                    loss_list.append(loss)

                if print_text:
                    print(line)

        except subprocess.CalledProcessError as e:
            print(f"エラー: 実行{i+1}回目に失敗しました\n{e.stderr}")

    return accuracy_list, loss_list, all_outputs


# 実行するPythonファイルのパス
file_path_py = r"neural_network.py"
# 実行回数 この回数実行して平均をとる
num_executions=10
# 学習回数のリスト
epochs_list=[1,2,3,4,5,6,7,8,9,10,20,30,40,50,60,70,80,90,100,200,300,400,500,600,700,800,900,1000,2000,3000,4000,5000,6000,7000,8000,9000,10000]
# epochs_list=[1,3]

# データを格納しているExcelファイル
file_path_excel = 'data/data_20項目_female_2122.xlsx'
# 全体のデータ数に対する学習用データ数の割合
train_data_ratio=0.8

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
    else:
        print("正解率または損失値の取得に失敗しました。")
