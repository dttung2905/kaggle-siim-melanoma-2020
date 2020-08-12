from engine import train, predict
import pandas as pd
import wandb

if __name__ == "__main__":
    #train(1)
    #train(2)
    #train(3)
    #train(4)
    #train(0)
    wandb.init(
       project="siim2020", entity="siim_melanoma", name="predict-rexnet2",
    )
    p1 = predict(1)
    p2 = predict(2)
    p3 = predict(3)
    p4 = predict(4)
    p0 = predict(0)

    sample = pd.read_csv(
       "../input/siim-isic-melanoma-classification/sample_submission.csv"
    )
    # sample.loc[:, "target"] = p0
    # sample.to_csv("../output/best/submission_0.csv", index=False)

    sample.loc[:, "target"] = (p1 + p2 + p3 + p4 + p0) / 5
    sample.to_csv("../output/submission_20200808.csv", index=False)
