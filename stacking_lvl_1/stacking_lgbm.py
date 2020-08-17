import pandas as pd
import numpy as np
from utils import get_meta_feature
import lightgbm as lgb
from sklearn.metrics import roc_auc_score


def get_column_name(meta_dict):
    result = []
    for k in meta_dict:
        for i in range(len(meta_dict[k])):
            result.append(k + "_" + meta_dict[k][i])

    return result


if __name__ == "__main__":
    params = {
        "boosting_type": "gbdt",
        "objective": "binary",
        #'metric': 'auc',
        "metric": "binary_logloss",
        #'max_depth': 13,
        #'num_leaves': 26,
        "max_depth": 15,
        "num_leaves": 30,
        "learning_rate": 0.0002,
        "feature_fraction": 0.8,
        "bagging_fraction": 0.8,
        "bagging_freq": 5,
        "verbose": 1,
        "seed": 5432,
    }
    agg_feature = {
        "age_approx": ["min", "max", "std", "count"],
        "predictions_1": ["min", "max", "std", "mean",],
        "predictions_2": ["min", "max", "std", "mean",],
        "predictions_3": ["min", "max", "std", "mean",],
        "predictions_4": ["min", "max", "std", "mean",],
        "predictions_5": ["min", "max", "std", "mean",],
        "predictions_6": ["min", "max", "std", "mean",],
        "predictions_7": ["min", "max", "std", "mean",],
        "predictions_8": ["min", "max", "std", "mean",],
    }
    original_train_df = pd.read_csv("../input/train_folds_leak_free.csv")

    # oof file tung
    khanh_1 = pd.read_csv("../input/stacking/best/khanh/oof_B5_384_929.csv")
    khanh_1 = khanh_1[["image_name", "predictions"]]
    khanh_1.columns = ["image_name", "predictions_1"]

    khanh_2 = pd.read_csv("../input/stacking/best/khanh/oof_B6_384_928.csv")
    khanh_2 = khanh_2[["image_name", "predictions"]]
    khanh_2.columns = ["image_name", "predictions_2"]

    nvnn_3 = pd.read_csv("../input/stacking/best/nvnn/Ens_v5_TTA8_train_oof.csv")
    nvnn_3 = nvnn_3[["image_name", "predictions"]]
    nvnn_3.columns = ["image_name", "predictions_3"]

    nvnn_4 = pd.read_csv("../input/stacking/best/nvnn/oof_b0_926_384_aedata.csv")
    nvnn_4 = nvnn_4[["image_name", "predictions"]]
    nvnn_4.columns = ["image_name", "predictions_4"]

    nvnn_5 = pd.read_csv("../input/stacking/best/nvnn/oof_b0_927.csv")
    nvnn_5 = nvnn_5[["image_name", "predictions"]]
    nvnn_5.columns = ["image_name", "predictions_5"]

    nvnn_6 = pd.read_csv("../input/stacking/best/nvnn/oof_b0_927v1.csv")
    nvnn_6 = nvnn_6[["image_name", "predictions"]]
    nvnn_6.columns = ["image_name", "predictions_6"]

    nvnn_7 = pd.read_csv("../input/stacking/best/nvnn/oof_b3_926_384.csv")
    nvnn_7 = nvnn_7[["image_name", "predictions"]]
    nvnn_7.columns = ["image_name", "predictions_7"]

    nvnn_8 = pd.read_csv("../input/stacking/best/nvnn/oof_b3_930.csv")
    nvnn_8 = nvnn_8[["image_name", "predictions"]]
    nvnn_8.columns = ["image_name", "predictions_8"]

    # test file
    original_test_df = pd.read_csv(
        "../input/siim-isic-melanoma-classification/test.csv"
    )
    test_khanh_1 = pd.read_csv("../input/stacking/best/khanh/B5_384_929.csv")
    test_khanh_1 = test_khanh_1[["image_name", "target"]]
    test_khanh_1.columns = ["image_name", "predictions_1"]

    test_khanh_2 = pd.read_csv("../input/stacking/best/khanh/B6_384_928.csv")
    test_khanh_2 = test_khanh_2[["image_name", "target"]]
    test_khanh_2.columns = ["image_name", "predictions_2"]

    test_nvnn_3 = pd.read_csv("../input/stacking/best/nvnn/Ens_v5_TTA8.csv")
    test_nvnn_3 = test_nvnn_3[["image_name", "target"]]
    test_nvnn_3.columns = ["image_name", "predictions_3"]

    test_nvnn_4 = pd.read_csv(
        "../input/stacking/best/nvnn/submission_b0_926_384_aedata.csv"
    )
    test_nvnn_4 = test_nvnn_4[["image_name", "target"]]
    test_nvnn_4.columns = ["image_name", "predictions_4"]

    test_nvnn_5 = pd.read_csv("../input/stacking/best/nvnn/submission_b0_927.csv")
    test_nvnn_5 = test_nvnn_5[["image_name", "target"]]
    test_nvnn_5.columns = ["image_name", "predictions_5"]

    test_nvnn_6 = pd.read_csv("../input/stacking/best/nvnn/submission_b0_927v1.csv")
    test_nvnn_6 = test_nvnn_6[["image_name", "target"]]
    test_nvnn_6.columns = ["image_name", "predictions_6"]

    test_nvnn_7 = pd.read_csv("../input/stacking/best/nvnn/submission_b3_926_384.csv")
    test_nvnn_7 = test_nvnn_7[["image_name", "target"]]
    test_nvnn_7.columns = ["image_name", "predictions_7"]

    test_nvnn_8 = pd.read_csv("../input/stacking/best/nvnn/submission_b3_930.csv")
    test_nvnn_8 = test_nvnn_8[["image_name", "target"]]
    test_nvnn_8.columns = ["image_name", "predictions_8"]

    # train_b6_384 = pd.read_csv("../input/stacking/train_EfficientNet_B6_384.csv")
    # train_b6_384 = train_b6_384[["image_name", "predictions"]]
    # train_b6_384.columns = ["image_name", "predictions_2"]

    # train_b4_256 = pd.read_csv("../input/stacking/train_EfficientNet_B4_256.csv")
    # train_b4_256 = train_b4_256[["image_name", "predictions"]]
    # train_b4_256.columns = ["image_name", "predictions_3"]

    # test_b4_256 = pd.read_csv("../input/stacking/test_EfficientNet_B6_256.csv")

    original_train_df = original_train_df.merge(
        khanh_1, left_on="image_name", right_on="image_name", how="left"
    )
    original_train_df = original_train_df.merge(
        khanh_2, left_on="image_name", right_on="image_name", how="left"
    )
    original_train_df = original_train_df.merge(
        nvnn_3, left_on="image_name", right_on="image_name", how="left"
    )
    original_train_df = original_train_df.merge(
        nvnn_4, left_on="image_name", right_on="image_name", how="left"
    )
    original_train_df = original_train_df.merge(
        nvnn_5, left_on="image_name", right_on="image_name", how="left"
    )
    original_train_df = original_train_df.merge(
        nvnn_6, left_on="image_name", right_on="image_name", how="left"
    )
    original_train_df = original_train_df.merge(
        nvnn_7, left_on="image_name", right_on="image_name", how="left"
    )
    original_train_df = original_train_df.merge(
        nvnn_8, left_on="image_name", right_on="image_name", how="left"
    )

    original_test_df = original_test_df.merge(
        test_khanh_1, left_on="image_name", right_on="image_name", how="left"
    )
    original_test_df = original_test_df.merge(
        test_khanh_2, left_on="image_name", right_on="image_name", how="left"
    )
    original_test_df = original_test_df.merge(
        test_nvnn_3, left_on="image_name", right_on="image_name", how="left"
    )
    original_test_df = original_test_df.merge(
        test_nvnn_4, left_on="image_name", right_on="image_name", how="left"
    )
    original_test_df = original_test_df.merge(
        test_nvnn_5, left_on="image_name", right_on="image_name", how="left"
    )
    original_test_df = original_test_df.merge(
        test_nvnn_6, left_on="image_name", right_on="image_name", how="left"
    )
    original_test_df = original_test_df.merge(
        test_nvnn_7, left_on="image_name", right_on="image_name", how="left"
    )
    original_test_df = original_test_df.merge(
        test_nvnn_8, left_on="image_name", right_on="image_name", how="left"
    )

    original_train_df["oof_preds"] = 0
    # original_train_df = original_train_df.head(100)
    # original_test_df = original_test_df.head(100)

    print("processing test feature")
    original_test_df, meta_columns = get_meta_feature(original_test_df)
    original_test_df_feature = original_test_df.groupby("patient_id").agg(agg_feature)
    # rename
    test_col_name = get_column_name(agg_feature)
    original_test_df_feature.columns = test_col_name
    original_test_df_feature = original_test_df_feature.reset_index(drop=False)
    original_test_df = original_test_df.merge(
        original_test_df_feature,
        left_on="patient_id",
        right_on="patient_id",
        how="left",
    )

    oof_preds = np.zeros(original_train_df.shape)

    sub_preds = np.zeros((original_test_df.shape[0],))
    final_auc = 0
    for fold in range(5):
        print(f"Training fold {fold}")
        df_train = original_train_df[original_train_df.kfold != fold].reset_index(
            drop=True
        )
        df_valid = original_train_df[original_train_df.kfold == fold].reset_index(
            drop=True
        )

        print("preprocessing data")
        df_train, meta_columns = get_meta_feature(df_train)
        df_valid, _ = get_meta_feature(df_valid)
        df_train_feature = df_train.groupby("patient_id").agg(agg_feature)
        df_valid_feature = df_valid.groupby("patient_id").agg(agg_feature)
        # rename
        col_name = get_column_name(agg_feature)
        df_train_feature.columns = col_name
        df_valid_feature.columns = col_name

        df_train_feature = df_train_feature.reset_index(drop=False)
        df_valid_feature = df_valid_feature.reset_index(drop=False)
        df_train = df_train.merge(
            df_train_feature, left_on="patient_id", right_on="patient_id", how="left"
        )
        df_valid = df_valid.merge(
            df_valid_feature, left_on="patient_id", right_on="patient_id", how="left"
        )
        meta_columns = [
            i
            for i in df_train.columns
            if i
            not in [
                "image_name",
                "patient_id",
                "anatom_site_general_challenge",
                "diagnosis",
                "benign_malignant",
                "target",
                "tfrecord",
                "width",
                "height",
                "patient_code",
                "kfold",
                "oof_preds",
            ]
        ]
        print(f"number of meta_columns is {len(meta_columns)}")
        print("Start training")

        lgb_train = lgb.Dataset(df_train[meta_columns], df_train["target"])
        lgb_eval = lgb.Dataset(df_valid[meta_columns], df_valid["target"])

        gbm = lgb.train(
            params,
            lgb_train,
            num_boost_round=100000,
            valid_sets=lgb_eval,
            early_stopping_rounds=20000,
            verbose_eval=3000,
        )
        original_train_df.loc[
            original_train_df.kfold == fold, "oof_preds"
        ] = gbm.predict(
            df_valid[meta_columns], num_iteration=gbm.best_iteration
        )  # get oof prediction
        # get best auc
        single_auc = roc_auc_score(
            original_train_df[original_train_df["kfold"] == fold]["target"],
            gbm.predict(df_valid[meta_columns], num_iteration=gbm.best_iteration),
        )  # get oof prediction )
        print(f"single fold AUC score : {single_auc}")
        final_auc += single_auc / 5
        # predict on test set, take average
        sub_preds += (
            gbm.predict(
                original_test_df[meta_columns], num_iteration=gbm.best_iteration
            )
            / 5
        )
        # tung = gbm.predict(original_test_df[meta_columns], num_iteration=gbm.best_iteration) / 5
        # print(tung.shape)
        # print(tung)
        # save the feature important
        # fold_importance_df = pd.DataFrame()
        # fold_importance_df["feature"] = feature_name
        # fold_importance_df["importance"] = np.log1p(gbm.feature_importance(
        #    importance_type='gain',
        #    iteration=gbm.best_iteration))
        # fold_importance_df["fold"] = n_fold + 1
        # feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    roc_auc_score = roc_auc_score(
        original_train_df["target"], original_train_df["oof_preds"]
    )
    print("_____________________________________")
    print(roc_auc_score)
    print(final_auc)

    submission = pd.read_csv(
        "../input/siim-isic-melanoma-classification/sample_submission.csv"
    )
    submission["target"] = sub_preds

    submission.to_csv(f"../output/best/{final_auc}_stacking.csv", index=False)
