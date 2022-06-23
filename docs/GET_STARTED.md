<img src="../figs/logo.png" align="right" width="25%">

# Getting Started

### Clone the GitHub Repo
```shell
git clone https://github.com/ldkong1205/PointCloud-C.git
cd PointCloud-C
```

### Set Up the Environment

```shell
conda create --name pointcloud-c python=3.7.5
conda activate pointcloud-c
pip install -r requirements.txt
cd SimpleView/pointnet2_pyt && pip install -e . && cd -
pip install -e pointcloudc_utils
```

### Download Pretrained Models

Please download existing pretrained models by
```shell
gdown https://drive.google.com/uc?id=11RONLZGg0ezxC16n57PiEZouqC5L0b_h
unzip pretrained_models.zip
```
Alternatively, you may download [pretrained models](https://drive.google.com/file/d/11RONLZGg0ezxC16n57PiEZouqC5L0b_h/view?usp=sharing) manually and extract it under root directory.
