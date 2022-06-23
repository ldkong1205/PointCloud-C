CUDA_VISIBLE_DEVICES=4 python train.py \
    --ckpts /mnt/lustre/ldkong/models/Point-MAE/segmentation/models/pretrain.pth \
    --root path/to/data \
    --learning_rate 0.0002 \
    --epoch 300 \
    --gpu 4