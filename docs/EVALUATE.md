<img src="../figs/logo.png" align="right" width="25%">

# Evaluation Script

### Outline

- [Classification](#classification)
- [Part Segmentation](#part-segmentation)


### Classification

#### Architecture

- DGCNN
```shell
python SimpleView/main.py --entry test_corrupt --model-path pretrained_models/DGCNN.pth --exp-config SimpleView/configs/dgcnn_dgcnn_run_1.yaml
```
- PointNet
```shell
python SimpleView/main.py --entry test_corrupt --model-path pretrained_models/PointNet.pth --exp-config SimpleView/configs/dgcnn_pointnet_run_1.yaml
```
- PointNet++
```shell
python SimpleView/main.py --entry test_corrupt --model-path pretrained_models/PointNet2.pth --exp-config SimpleView/configs/dgcnn_pointnet2_run_1.yaml
```
- RSCNN
```shell
python SimpleView/main.py --entry test_corrupt --model-path pretrained_models/RSCNN.pth --exp-config SimpleView/configs/dgcnn_rscnn_run_1.yaml
```
- SimpleView
```shell
python SimpleView/main.py --entry test_corrupt --model-path pretrained_models/SimpleView.pth --exp-config SimpleView/configs/dgcnn_simpleview_run_1.yaml
```
- PCT
```shell
python PCT/main.py --exp_name=test --num_points=1024 --use_sgd=True --eval_corrupt=True --model_path pretrained_models/PCT.t7 --test_batch_size 8 --model PCT
```
- GDANet
```shell
python GDANet/main_cls.py --eval_corrupt=True --model_path pretrained_models/GDANet.t7
```
- PAConv
```shell
python PAConv/obj_cls/main.py --config PAConv/obj_cls/config/dgcnn_paconv_test.yaml --model_path ../pcdrobustness/pretrained_models/PAConv.t7 --eval_corrupt True
```
- CurveNet
```shell
python3 CurveNet/core/main_cls.py --exp_name=test --eval_corrupt=True --model_path pretrained_models/CurveNet.t7
```
- RPC
```shell
python PCT/main.py --exp_name=test --num_points=1024 --use_sgd=True --eval_corrupt=True --model_path pretrained_models/RPC.t7 --test_batch_size 8 --model RPC
```

#### Augmentation

- DGCNN + PointWOLF
```shell
python PointWOLF/main.py --exp_name=test --model=dgcnn --num_points=1024 --k=20 --use_sgd=True --eval_corrupt=True --model_path pretrained_models/DGCNN_PointWOLF.t7
```
- DGCNN + RSMix
```shell
python PointWOLF/main.py --exp_name=test --model=dgcnn --num_points=1024 --k=20 --use_sgd=True --eval_corrupt=True --model_path pretrained_models/DGCNN_RSMix.t7
```
- DGCNN + WOLFMix
```shell
python PointWOLF/main.py --exp_name=test --model=dgcnn --num_points=1024 --k=20 --use_sgd=True --eval_corrupt=True --model_path pretrained_models/DGCNN_WOLFMix.t7
```
- GDANet + WOLFMix
```shell
python GDANet/main_cls.py --eval_corrupt=True --model_path pretrained_models/GDANet_WOLFMix.t7
```
- RPC + WOLFMix (final)
```shell
python PCT/main.py --exp_name=test --num_points=1024 --use_sgd=True --eval_corrupt=True --model_path pretrained_models/RPC_WOLFMix_final.t7 --test_batch_size 8 --model RPC
```


### Part Segmentation

Coming soon.




