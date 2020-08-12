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
    'boosting_type': 'gbdt',
    'objective': 'binary',
    'metric': 'auc',
    'max_depth': 10,
    'num_leaves': 20,
    'learning_rate': 0.0002,
    'feature_fraction': 0.9,
    'bagging_fraction': 0.8,
    'bagging_freq': 5,
    'verbose': 1,
    }
    agg_feature = {
        "age_approx" : ["min", "max", "std", "count"],
        "predictions_1": ["min", "max", "std", "mean", ],
        "predictions_2": ["min", "max", "std", "mean", ],
        "predictions_3": ["min", "max", "std", "mean", ],
    }
    original_train_df = pd.read_csv("../input/train_folds_leak_free.csv")

    train_b6_256 = pd.read_csv("../input/stacking/train_EfficientNet_B6_256.csv")
    train_b6_256 = train_b6_256[["image_name", "predictions"]]
    train_b6_256.columns = ["image_name", "predictions_1"]

    train_b6_384 = pd.read_csv("../input/stacking/train_EfficientNet_B6_384.csv")
    train_b6_384 = train_b6_384[["image_name", "predictions"]]
    train_b6_384.columns = ["image_name", "predictions_2"]

    train_b4_256 = pd.read_csv("../input/stacking/train_EfficientNet_B4_256.csv")
    train_b4_256 = train_b4_256[["image_name", "predictions"]]
    train_b4_256.columns = ["image_name", "predictions_3"]

    test_b4_256 = pd.read_csv("../input/stacking/test_EfficientNet_B6_256.csv")
    original_train_df = original_train_df.merge(train_b6_256, left_on='image_name', right_on="image_name", how="left")
    original_train_df = original_train_df.merge(train_b6_384, left_on='image_name', right_on="image_name", how="left")
    original_train_df = original_train_df.merge(train_b4_256, left_on='image_name', right_on="image_name", how="left")
    print(roc_auc_score(original_train_df["target"], original_train_df["predictions_1"]))
    print(roc_auc_score(original_train_df["target"], original_train_df["predictions_2"]))
    print(roc_auc_score(original_train_df["target"], original_train_df["predictions_3"]))
    #oof_preds = np.zeros(original_train_df.shape)
    #original_train_df["oof_preds"] = 0
    ##original_train_df = original_train_df.head(100)
    #original_train_df_feature = original_train_df.groupby("patient_id").agg(agg_feature)
    #for fold in range(5):
    #    print(f"Training fold {fold}")
    #    df_train = original_train_df[original_train_df.kfold != fold].reset_index(drop=True)
    #    df_valid = original_train_df[original_train_df.kfold == fold].reset_index(drop=True)

    #    print("preprocessing data")
    #    df_train, meta_columns = get_meta_feature(df_train)
    #    df_valid, _ = get_meta_feature(df_valid)
    #    df_train_feature = df_train.groupby("patient_id").agg(agg_feature)
    #    df_valid_feature = df_valid.groupby("patient_id").agg(agg_feature)
    #    # rename
    #    col_name = get_column_name(agg_feature)
    #    df_train_feature.columns = col_name
    #    df_valid_feature.columns = col_name

    #    df_train_feature = df_train_feature.reset_index(drop=False) 
    #    df_valid_feature = df_valid_feature.reset_index(drop=False) 
    #    df_train = df_train.merge(df_train_feature, left_on="patient_id", right_on="patient_id", how="left")
    #    df_valid = df_valid.merge(df_valid_feature, left_on="patient_id", right_on="patient_id", how="left")
    #    #meta_columns = meta_columns + ["predictions_1", "predictions_2", "predictions_3"] + col_name
    #    meta_columns = [i for i in df_train.columns if i not in ['image_name', 'patient_id', 
    #                                                             'anatom_site_general_challenge', 'diagnosis', 'benign_malignant',
    #                                                             'target', 'tfrecord', 'width', 'height', 'patient_code', 'kfold']]
    #    print(f"number of meta_columns is {len(meta_columns)}")
    #    print("Start training")

    #    lgb_train = lgb.Dataset(df_train[meta_columns], df_train["target"])
    #    lgb_eval = lgb.Dataset(df_valid[meta_columns], df_valid["target"])

    #    gbm = lgb.train(params,
    #                lgb_train,
#   #              num_boost_round = 2,
    #                num_boost_round=10000,
    #                valid_sets=lgb_eval,
    #                early_stopping_rounds=4000,
    #                verbose_eval=500)
    #    original_train_df.loc[original_train_df.kfold==fold, "oof_preds"] = gbm.predict(df_valid[meta_columns], num_iteration=gbm.best_iteration) #get oof prediction

    #    #predict on test set, take average
    #    #sub_preds += gbm.predict(X_test[feature_name], num_iteration=gbm.best_iteration) / folds.n_splits 
    #    #save the feature important 
    #    #fold_importance_df = pd.DataFrame()
    #    #fold_importance_df["feature"] = feature_name
    #    #fold_importance_df["importance"] = np.log1p(gbm.feature_importance(
    #    #    importance_type='gain',
    #    #    iteration=gbm.best_iteration))
    #    #fold_importance_df["fold"] = n_fold + 1
    #    #feature_importance_df = pd.concat([feature_importance_df, fold_importance_df], axis=0)

    #roc_auc_score = roc_auc_score(original_train_df["target"], original_train_df["oof_preds"])
    #print("_____________________________________")
    #print(roc_auc_score)
