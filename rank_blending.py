import pandas as pd

if __name__ == "__main__":
    sub_stacking1 = pd.read_csv("../output/best/best_sub/9393_ensemble_nghia.csv")
    sub_stacking2 = pd.read_csv(
        "../output/best/best_sub/0.9435202868760748_stacking.csv"
    )
    a = pd.Series(sub_stacking1["target"].values)
    b = pd.Series(sub_stacking2["target"].values).rank(pct=True).values

    sub = pd.read_csv(
        "../input/siim-isic-melanoma-classification/sample_submission.csv"
    )
    sub["target"] = (a + b) / 2
    sub.to_csv("../output/best/best_sub/final_ensemble_submission.csv", index=False)
