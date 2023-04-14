<img src="../figs/logo.png" align="right" width="25%">

# Prepare Data

### Classification
Download `ModelNet-C` by:
```shell
cd data
gdown https://drive.google.com/uc?id=1KE6MmXMtfu_mgxg4qLPdEwVD5As8B0rm
unzip modelnet_c.zip && cd ..
```
Alternatively, you may download `ModelNet-C` from our <a href="https://pointcloud-c.github.io/download.html" target='_blank'>project page</a> or run the following command:
```shell
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1KE6MmXMtfu_mgxg4qLPdEwVD5As8B0rm' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1KE6MmXMtfu_mgxg4qLPdEwVD5As8B0rm" -O modelnet_c.zip && rm -rf /tmp/cookies.txt
```

### Part Segmentation
Download `ShapeNet-C` by:
```shell
cd data
gdown https://drive.google.com/uc?id=1OqdJ9q1GjVu46QdPC4coa8M5e-xOrJ_g
unzip shapenet_c.zip && cd ..
```
Alternatively, you may download `ShapeNet-C` from our <a href="https://pointcloud-c.github.io/download.html" target='_blank'>project page</a> or run the following command:
```shell
wget --load-cookies /tmp/cookies.txt "https://docs.google.com/uc?export=download&confirm=$(wget --quiet --save-cookies /tmp/cookies.txt --keep-session-cookies --no-check-certificate 'https://docs.google.com/uc?export=download&id=1OqdJ9q1GjVu46QdPC4coa8M5e-xOrJ_g' -O- | sed -rn 's/.*confirm=([0-9A-Za-z_]+).*/\1\n/p')&id=1OqdJ9q1GjVu46QdPC4coa8M5e-xOrJ_g" -O shapenet_c.zip && rm -rf /tmp/cookies.txt
```


### Dataset Structure
```
root
 └─── dataset_c
         └───── add_global_0.h5
         └───── ...
         └───── add_local_0.h5
         └───── ...
         └───── dropout_global_0.h5
         └───── ...
         └───── dropout_local_0.h5
         └───── ...
         └───── jitter_0.h5
         └───── ...
         └───── rotate_0.h5
         └───── ...
         └───── scale_0.h5
         └───── ...
         └───── clean.h5
 └─── README.txt
 ```
 
 
### License
 
This benchmark is made freely available to academic and non-academic entities for non-commercial purposes such as academic research, teaching, scientific publications, or personal experimentation. Permission is granted to use the data given that you agree:

- That the benchmark comes “AS IS”, without express or implied warranty. Although every effort has been made to ensure accuracy, we do not accept any responsibility for errors or omissions.
- That you may not use the benchmark or any derivative work for commercial purposes as, for example, licensing or selling the data, or using the data with a purpose to procure a commercial gain.
- That you include a reference to PointCloud-C (including ModelNet-C, ShapeNet-C, and the specially generated data for academic challenges) in any work that makes use of the benchmark. For research papers, please cite our preferred publications as listed on our webpage.

