# RSMix for PointNet++(TensorFlow)
We utilize the original released codes of [PointNet++](https://github.com/charlesq34/pointnet2/), which is implemented with TensorFlow.

* RSMix is implemented in `rsmix_provider.py`.

## Prepare Dataset
* You can get sampled point clouds of ModelNet40 (XYZ and normal from mesh, 10k points per shape) <a href="https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip">here (1.6GB)</a>. 

Move the uncompressed data folder or create symbolic link to `data/modelnet40_normal_resampled`.

* You can also get sampled point clouds of ModelNet40 (XYZ and normal from mesh, 10k points per shape) as hdf5 format <a href="https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip">here (435MB)</a>.

Move the uncompressed data folder or create symbolic link to `data/modelnet40_ply_hdf5_2048`.

## Environment
Follow the environment setting of original `PointNet++` code.

[[PointNet++ Environment setting]](https://github.com/charlesq34/pointnet2/blob/master/README.md)

## Point Cloud Classification
### Training
#### Run the training script:

```
python train.py --model pointnet2_cls_ssg --log_dir ./log/$exp_name $conventional_data_augmentation_arguements $dataset_arguments --beta 1.0
```
--> `--beta 1.0` : RSMix argument

* conventional data agumentation arguments:

(if you use these arguments, input the argument `--convda` in advance)

`--shuffle` : Random shuffle augmentation

`--jitter` : Jitter augmentation

`--rot` : Random Rotation augmentation

`--rdscale` : Random Scaling augmentation

`--shift` : Random Shif augmentation

* RandDrop augmentation argument:

`--rddrop` : RandDrop augmentation

* Dataset_argument:

(Default dataset: modelnet40)

`--modelnet10`: modelnet10 argument

* For PointNet training:

replace the model name as `pointnet_cls_rsmix`



### Evaluation

#### Run the evaluation script:

**PointNet++**

For ModelNet40 evaluation,

- Single-view evaluation:
```
python evaluate.py --num_votes 1 --model_path ./log/$exp_name/model.ckpt
```

- Multi-view evaluation:
```
python evaluate.py --num_votes 12 --model_path ./log/$exp_name/model.ckpt
```

For ModelNet10 evaluation,

- Single-view evaluation:
```
python evaluate_modelnet10.py --num_votes 1 --model_path ./log/$exp_name/model.ckpt
```

- Multi-view evaluation:
```
python evaluate_modelnet10.py --num_votes 12 --model_path ./log/$exp_name/model.ckpt
```


**PointNet**

For ModelNet40 Evaluation,

- Single-view evaluation:
```
python evaluate.py --num_votes 1 --model pointnet_cls_basic --model_path ./log/$exp_name/model.ckpt
```

- Multi-view evaluation:
```
python evaluate.py --num_votes 12 --model pointnet_cls_basic --model_path ./log/$exp_name/model.ckpt
```





### Save and Visualize Samples

You can input additional arguments related to the other augmentations if you want.

- Save mixed samples:
```
python train_data_mix_save.py  --model pointnet2_cls_ssg --log_dir $log_dir --mixed_data_dir ./data_mix --mixed_data_save --beta 1.0 $additional_augmentation_arguments
```

- Visualize Samples:
```
python ./utils/show3d_balls_rsmix.py  --background_white --ball_mix(if you  want knn visualize, use --knn_mix) --path $mixed_data 
```
We utilize the viusalization tool in released code of [PointNet++](https://github.com/charlesq34/pointnet2/).

The original code for visualization is <a href="http://github.com/fanhqme/PointSetGeneration">here</a>