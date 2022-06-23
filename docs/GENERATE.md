<img src="../figs/logo.png" align="right" width="25%">

# Generation Your Own Corruption Sets

You may generate more "PointCloud-C" sets by:
```shell
python build/corrupt.py
```

:warning: Note that the script uses a **different** random seed from the official ModelNet-C and ShapeNet-C.
One should NOT report results on self-generated corruption sets.
