# RSMix for DGCNN(PyTorch)
We utilize the original released codes of [DGCNN](https://github.com/WangYueFt/dgcnn/tree/master/pytorch), which is implemented with PyTorch.

* RSMix is implemented in `rsmix_provider.py`.

## Prepare Dataset
* You can get sampled point clouds of ModelNet40 (XYZ and normal from mesh, 10k points per shape) <a href="https://shapenet.cs.stanford.edu/media/modelnet40_normal_resampled.zip">here (1.6GB)</a>. 

Move the uncompressed data folder or create symbolic link to `data/modelnet40_normal_resampled`.

* You can also get sampled point clouds of ModelNet40 (XYZ and normal from mesh, 10k points per shape) as hdf5 format <a href="https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip">here (435MB)</a>.

Move the uncompressed data folder or create symbolic link to `data/modelnet40_ply_hdf5_2048`.

## Environment
Follow the environment setting of original `DGCNN` code.

[[DGCNN PyTorch]](https://github.com/WangYueFt/dgcnn)


## Point Cloud Classification
### Training
#### Run the training script(epoch=500):

* 1024 points
```
python main.py --exp_name=rsmix_dgcnn_1024 --model=dgcnn --num_points=1024 --k=20 --use_sgd=True --beta 1.0 --epochs 500
```

* 2048 points
```
python main.py --exp_name=rsmix_dgcnn_2048 --model=dgcnn --num_points=2048 --k=40 --use_sgd=True --beta 1.0 --epochs 500
```

Note: if you want to test the combinations of augmentations with RSMix,

you can selectively input the augmentation-related arguments from one of the follow arguments.

* conventional data agumentation arguments:

`--shuffle` : Random shuffle augmentation

`--jitter` : Jitter augmentation

`--rot` : Random Rotation augmentation

`--rdscale` : Random Scaling augmentation

`--shift` : Random Shif augmentation

* RandDrop augmentation argument:

`--rddrop` : RandDrop augmentation

Additionally, if you want to test with ModelNet10, please input the argument `--modelnet10`.
Default dataset is ModelNet40.


### Evaluation 

#### Run the evaluation script after training finished:

* 1024 points
```
python main.py --exp_name=rsmix_dgcnn_1024_eval --model=dgcnn --num_points=1024 --k=20 --use_sgd=True --eval=True --model_path=checkpoints/rsmix_dgcnn_1024/models/model.t7
```

* 2048 points
```
python main.py --exp_name=rsmix_dgcnn_2048_eval --model=dgcnn --num_points=2048 --k=40 --use_sgd=True --eval=True --model_path=checkpoints/rsmix_dgcnn_2048/models/model.t7
```



### Evaluation with pretrained model
#### Run the evaluation script with pretrained models:

* 1024 points
```
python main.py --exp_name=rsmix_dgcnn_1024_eval --model=dgcnn --num_points=1024 --k=20 --use_sgd=True --eval=True --model_path=pretrained/model.1024.t7
```

* 2048 points
```
python main.py --exp_name=rsmix_dgcnn_2048_eval --model=dgcnn --num_points=2048 --k=40 --use_sgd=True --eval=True --model_path=pretrained/model.2048.t7
```
