python train.py \
	--gpu 1,5,6,7 \
	--use_sgd \
	--xavier_init \
	--scheduler cos \
	--model dgcnn_partseg \
	--log_dir occo_dgcnn \
	--batch_size 16 \
	--restore \
	--restore_path /mnt/lustre/ldkong/models/OcCo/OcCo_Torch/pretrain/dgcnn_occo_seg.pth