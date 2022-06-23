import os
import h5py
from torch.utils.data import Dataset

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../../data/pointcloud_c')   # pls change the data dir accordingly


def load_h5(h5_name):
    f = h5py.File(h5_name, 'r')
    data = f['data'][:].astype('float32')
    label = f['label'][:].astype('int64')
    f.close()
    return data, label


class ModelNetC(Dataset):
    def __init__(self, split):
        h5_path = os.path.join(DATA_DIR, split + '.h5')
        self.data, self.label = load_h5(h5_path)

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        return pointcloud, label

    def __len__(self):
        return self.data.shape[0]
