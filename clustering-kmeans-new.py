# kmeans_excel.py

import argparse
import numpy as np
import pandas as pd

from sklearn.preprocessing import StandardScaler
from sklearn.cluster import KMeans


def normalize_missing_value(x):
    """
    Excel内の欠損値、および 'missing' と書かれた値を NaN に変換する。
    クラスタリング用データの前処理にのみ使う。
    """
    if pd.isna(x):
        return np.nan

    if isinstance(x, str):
        value = x.strip()
        if value == "":
            return np.nan
        if value.lower() == "missing":
            return np.nan

    return x


def convert_sheet_name(sheet_name):
    """
    コマンドラインから --sheet_name 0 のように指定された場合、
    文字列 "0" ではなく整数 0 として扱う。
    """
    try:
        return int(sheet_name)
    except ValueError:
        return sheet_name


def main(data, result, sheet_name=0, n_clusters=3, random_state=42):
    sheet_name = convert_sheet_name(str(sheet_name))

    # Excelをヘッダーなしで読み込む
    df = pd.read_excel(
        data,
        sheet_name=sheet_name,
        header=None,
        engine="openpyxl"
    )

    if df.shape[0] < 3:
        raise ValueError(
            "Excelには少なくとも1行目=項目名、2行目=項目ID、3行目以降=データが必要です。"
        )

    if df.shape[1] < 2:
        raise ValueError(
            "説明変数と目的変数を含め、少なくとも2列以上が必要です。"
        )

    # 1行目: 項目名、2行目: 項目ID、3行目以降: データ
    item_names = df.iloc[0].tolist()
    item_ids = df.iloc[1].tolist()
    data = df.iloc[2:].reset_index(drop=True)

    if len(data) < n_clusters:
        raise ValueError(
            f"データ件数 {len(data)} 件に対して、クラスタ数 {n_clusters} は大きすぎます。"
        )

    # 最終列は目的変数としてクラスタリング対象から除外
    explanatory_data_original = data.iloc[:, :-1].copy()
    target_data_original = data.iloc[:, -1].copy()

    # クラスタリング用にのみ missing / 欠損値を NaN に統一
    explanatory_data_missing = explanatory_data_original.apply(
        lambda col: col.map(normalize_missing_value)
    )

    # k-meansは数値データ前提なので、数値に変換
    explanatory_numeric = explanatory_data_missing.apply(
        pd.to_numeric,
        errors="coerce"
    )

    # missing / 空欄以外で数値変換できなかった値を検出
    invalid_mask = explanatory_numeric.isna() & explanatory_data_missing.notna()

    if invalid_mask.any().any():
        invalid_cols = []

        for col in explanatory_data_missing.columns:
            if invalid_mask[col].any():
                excel_col_no = col + 1
                col_name = item_names[col]
                invalid_cols.append(f"{excel_col_no}列目: {col_name}")

        raise ValueError(
            "数値に変換できない説明変数があります。"
            "k-meansではカテゴリ値をそのまま扱えません。\n"
            + "\n".join(invalid_cols)
        )

    # 平均補完
    col_means = explanatory_numeric.mean(axis=0)

    # 全て欠損の列は平均を計算できないためエラー
    all_missing_cols = col_means[col_means.isna()].index.tolist()

    if all_missing_cols:
        cols = []

        for col in all_missing_cols:
            excel_col_no = col + 1
            col_name = item_names[col]
            cols.append(f"{excel_col_no}列目: {col_name}")

        raise ValueError(
            "全て欠損している説明変数があります。平均補完できません。\n"
            + "\n".join(cols)
        )

    explanatory_imputed = explanatory_numeric.fillna(col_means)

    # k-means実行前に標準化
    scaler = StandardScaler()
    X_scaled = scaler.fit_transform(explanatory_imputed)

    # k-means実行
    kmeans = KMeans(
        n_clusters=n_clusters,
        random_state=random_state,
        n_init=10
    )

    labels_zero_based = kmeans.fit_predict(X_scaled)

    # クラスタ番号を1始まりにする
    labels = labels_zero_based + 1

    # コンソール出力
    print("k-meansクラスタリングが完了しました。")
    print(f"クラスタ数: {n_clusters}")

    print("各クラスタのデータ件数:")
    for cluster_no in range(1, n_clusters + 1):
        count = np.sum(labels == cluster_no)
        print(f"  クラスタ {cluster_no}: {count} 件")

    # 出力用データ
    # 注意:
    # クラスタリングには平均補完後データを使用するが、
    # Excel出力では補完前の説明変数データをそのまま出力する。
    output_data = pd.concat(
        [
            explanatory_data_original.reset_index(drop=True),
            pd.Series(labels, name="cluster"),
            target_data_original.reset_index(drop=True)
        ],
        axis=1
    )

    # 1行目: 項目名
    output_item_names = item_names[:-1] + ["クラスタ番号"] + [item_names[-1]]

    # 2行目: 項目ID
    output_item_ids = item_ids[:-1] + ["cluster_id"] + [item_ids[-1]]

    # Excel出力用に2段ヘッダーを復元
    output_df = pd.DataFrame(
        [output_item_names, output_item_ids] + output_data.values.tolist()
    )

    output_df.to_excel(
        result,
        index=False,
        header=False,
        engine="openpyxl"
    )

    print(f"出力ファイル: {result}")


if __name__ == "__main__":
    parser = argparse.ArgumentParser()

    parser.add_argument(
        "--data",
        help="入力Excelファイルパス"
    )

    parser.add_argument(
        "--result",
        help="出力Excelファイルパス"
    )

    parser.add_argument(
        "--sheet_name",
        default=0,
        help="読み込むシート名または番号。省略時は先頭シート"
    )

    parser.add_argument(
        "--n_clusters",
        type=int,
        default=3,
        help="k-meansのクラスタ数"
    )

    parser.add_argument(
        "--random_state",
        type=int,
        default=42,
        help="乱数シード"
    )

    args = parser.parse_args()

    main(
        data=args.data,
        result=args.result,
        sheet_name=args.sheet_name,
        n_clusters=args.n_clusters,
        random_state=args.random_state
    )