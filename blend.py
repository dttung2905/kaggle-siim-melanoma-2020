import pandas as pd


sub0 = pd.read_csv("../output/best/submission_0.csv")
sub1 = pd.read_csv("../output/best/submission_1.csv")
sub2 = pd.read_csv("../output/best/submission_2.csv")
sub3 = pd.read_csv("../output/best/submission_3.csv")
sub4 = pd.read_csv("../output/best/submission_4.csv")


sub0["target"] = (
    sub0["target"] + sub1["target"] + sub2["target"] + sub3["target"] + sub4["target"]
) / 5

sub0.to_csv("../output/best/blend.csv", index=False)
