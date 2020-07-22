import pandas as pd
import numpy as np
from sklearn import model_selection
import argparse
import random
from pathlib import Path
from collections import Counter, defaultdict


def get_args():
    parser = argparse.ArgumentParser()
    parser.add_argument("--source", type=str, required=True, help="source data")
    parser.add_argument("--type", type=str, required=True, help="split type")
    parser.add_argument("--target", type=str, required=True, help="target data")
    return parser.parse_args()


def stratified_group_k_fold(X, y, groups, k, seed=None):
    labels_num = np.max(y) + 1
    y_counts_per_group = defaultdict(lambda: np.zeros(labels_num))
    y_distr = Counter()
    for label, g in zip(y, groups):
        y_counts_per_group[g][label] += 1
        y_distr[label] += 1

    y_counts_per_fold = defaultdict(lambda: np.zeros(labels_num))
    groups_per_fold = defaultdict(set)

    def eval_y_counts_per_fold(y_counts, fold):
        y_counts_per_fold[fold] += y_counts
        std_per_label = []
        for label in range(labels_num):
            label_std = np.std(
                [y_counts_per_fold[i][label] / y_distr[label] for i in range(k)]
            )
            std_per_label.append(label_std)
        y_counts_per_fold[fold] -= y_counts
        return np.mean(std_per_label)

    groups_and_y_counts = list(y_counts_per_group.items())
    random.Random(seed).shuffle(groups_and_y_counts)

    for g, y_counts in sorted(groups_and_y_counts, key=lambda x: -np.std(x[1])):
        best_fold = None
        min_eval = None
        for i in range(k):
            fold_eval = eval_y_counts_per_fold(y_counts, i)
            if min_eval is None or fold_eval < min_eval:
                min_eval = fold_eval
                best_fold = i
        y_counts_per_fold[best_fold] += y_counts
        groups_per_fold[best_fold].add(g)

    all_groups = set(groups)
    for i in range(k):
        train_groups = all_groups - groups_per_fold[i]
        test_groups = groups_per_fold[i]

        train_indices = [i for i, g in enumerate(groups) if g in train_groups]
        test_indices = [i for i, g in enumerate(groups) if g in test_groups]

        yield train_indices, test_indices


def get_distribution(y_vals):
    y_distr = Counter(y_vals)
    y_vals_sum = sum(y_distr.values())
    return [f"{y_distr[i] / y_vals_sum:.5%}" for i in range(np.max(y_vals) + 1)]


if __name__ == "__main__":
    args = get_args()
    # create folds
    df = pd.read_csv(args.source)
    df["kfold"] = -1
    df = df.sample(frac=1).reset_index(drop=True)
    y = df.target.values

    assert args.type in ["group_k_fold", "stratified_k_fold", "stratified_group_k_fold"]
    if args.type == "group_k_fold":
        print("Using groupKFold")
        kf = model_selection.GroupKFold(n_splits=5)
        for f, (t_, v_) in enumerate(
            kf.split(X=df, y=y, groups=df["patient_id"].tolist())
        ):
            df.loc[v_, "kfold"] = f
        df.to_csv(args.target, index=False)
    elif args.type == "stratified_k_fold":
        print("Using StratifiedKFold")
        kf = model_selection.StratifiedKFold(n_splits=5)
        for f, (t_, v_) in enumerate(kf.split(X=df, y=y)):
            df.loc[v_, "kfold"] = f
        df.to_csv(args.target, index=False)

    elif args.type == "stratified_group_k_fold":
        print("Using stratified_group_k_fold")
        groups = np.array(df["patient_id"].values)
        for f, (t_, v_) in enumerate(
            stratified_group_k_fold(X=df, y=y, groups=groups, k=5, seed=1)
        ):
            df.loc[v_, "kfold"] = f

            distrs = [get_distribution(y)]
            index = ["training set"]
            dev_y, val_y = y[t_], y[v_]
            dev_groups, val_groups = groups[t_], groups[v_]
            # making sure that train and validation group do not overlap:
            assert len(set(dev_groups) & set(val_groups)) == 0

            distrs.append(get_distribution(dev_y))
            index.append(f"development set - fold {f}")
            distrs.append(get_distribution(val_y))
            index.append(f"validation set - fold {f}")

            output_statistic = pd.DataFrame(
                distrs,
                index=index,
                columns=[f"Label {l}" for l in range(np.max(y) + 1)],
            )
            print(output_statistic.head(20))

        df.to_csv(args.target, index=False)
