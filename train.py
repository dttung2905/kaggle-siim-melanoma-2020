from engine import train, predict
import pandas as pd
import wandb

if __name__ == "__main__":
    # train(1)
    # train(2)
    # train(3)
    # train(4)
    # train(0)
    wandb.init(
        project="siim2020", entity="siim_melanoma", name="test",
    )
    p1 = predict(1)
    # p2 = predict(2)
    # p3 = predict(3)
    # p4 = predict(4)
    # p0 = predict(0)

    sample = pd.read_csv(
        "../input/siim-isic-melanoma-classification/sample_submission.csv"
    )
    sample.loc[:, "target"] = p1
    sample.to_csv("../output/submission_20200721_test_1.csv", index=False)

    # sample.loc[:, "target"] = (p1 + p2 + p3 + p4 + p0) / 5
    # sample.to_csv("../output/submission_20200607_test.csv", index=False)
