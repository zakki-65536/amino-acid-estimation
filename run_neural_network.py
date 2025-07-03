import subprocess
import re

def execute_python_file(file_path, num_executions=10):
    accuracy_list = []
    loss_list = []
    all_outputs = []

    for i in range(num_executions):
        print(f"【実行{i+1}回目】")
        try:
            result = subprocess.run(
                ["python", file_path],
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

                if acc_match:
                    accuracy = float(acc_match.group(1))
                    accuracy_list.append(accuracy)

                if loss_match:
                    loss = float(loss_match.group(1))
                    loss_list.append(loss)

        except subprocess.CalledProcessError as e:
            print(f"エラー: 実行{i+1}回目に失敗しました\n{e.stderr}")

    return accuracy_list, loss_list, all_outputs


# 実行するPythonファイルのパスを指定（適宜変更）
file_path = r"E:\regression_project2\neural_network.py"


# 10回実行して結果を取得
accuracies, losses, outputs = execute_python_file(file_path, num_executions=10)

# 各実行の出力表示（任意）
for output in outputs:
    print(output)

# 結果集計と表示
if accuracies and losses:
    avg_accuracy = sum(accuracies) / len(accuracies)
    avg_loss = sum(losses) / len(losses)
    print("=== 集計結果（10回平均）===")
    print(f"平均正解率（±5%以内）: {avg_accuracy:.2f}%")
    print(f"平均学習損失値 (loss): {avg_loss:.4f}")
else:
    print("正解率または損失値の取得に失敗しました。")
