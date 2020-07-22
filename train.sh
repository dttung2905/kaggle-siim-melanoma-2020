LD_LIBRARY_PATH=/usr/local/cuda/lib64
#python3 train.py --config configs/resnet50.yaml
#python3 train.py --config configs/SEResnext50_32x4d.yaml
python3 train.py --config configs/efficientNetConfig_B0.yaml
#python3 train.py --config configs/efficientNetConfig_B6.yaml
