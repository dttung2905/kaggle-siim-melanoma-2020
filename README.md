# siim-melanoma-2020

repo for Kaggle competition

## 1. Preprocess data

Run this command to create fold
```
python3 make_folds.py \
  --source ../input/siim-isic-melanoma-classification/train.csv \
  --target ../input/train_folds.csv \
  --type group_k_fold
```

If you want to use 2019 ISIC data, first you have to get only the malignant images and resize it

```
python3 resize_image.py \
  --source2019 ../input/ISIC_2019_Training_GroundTruth.csv \
  --source2020 ../input/siim-isic-melanoma-classification/train.csv \
  --inputimgfolder ../input/2019_data/ISIC_2019_Training_Input/ISIC_2019_Training_Input/ \
  --outputimgfolder ../input/2019_data/isic_2019_v2
```

then preprocess the training csv
```
python3 preprocess_2019_data.py \
  --source ../input/ISIC_2019_Training_GroundTruth.csv \
  --target ../input/train_2019_supplements.csv
```

## 2. Run model
Run the following bash script
```
sh train.sh
```
