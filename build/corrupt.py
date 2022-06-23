import os
import glob
import h5py
import numpy as np
from corrupt_utils import corrupt_scale, corrupt_jitter, corrupt_rotate, corrupt_dropout_global, corrupt_dropout_local, \
    corrupt_add_global, corrupt_add_local

NUM_POINTS = 1024
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
DATA_DIR = os.path.join(BASE_DIR, '../data')

np.random.seed(0)

corruptions = {
    'clean': None,
    'scale': corrupt_scale,
    'jitter': corrupt_jitter,
    'rotate': corrupt_rotate,
    'dropout_global': corrupt_dropout_global,
    'dropout_local': corrupt_dropout_local,
    'add_global': corrupt_add_global,
    'add_local': corrupt_add_local,
}


def download():
    if not os.path.exists(DATA_DIR):
        os.mkdir(DATA_DIR)
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
        www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
        zipfile = os.path.basename(www)
        os.system('wget %s --no-check-certificate; unzip %s' % (www, zipfile))
        os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
        os.system('rm %s' % (zipfile))


def load_data(partition):
    download()
    all_data = []
    all_label = []
    for h5_name in glob.glob(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048', 'ply_data_%s*.h5' % partition)):
        f = h5py.File(h5_name, 'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        f.close()
        all_data.append(data)
        all_label.append(label)
    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_data = all_data[:, :NUM_POINTS, :]
    return all_data, all_label


def save_data(all_data, all_label, corruption_type, level):
    if not os.path.exists(os.path.join(DATA_DIR, 'modelnet_c')):
        os.makedirs(os.path.join(DATA_DIR, 'modelnet_c'))
    if corruption_type == 'clean':
        h5_name = os.path.join(DATA_DIR, 'modelnet_c', '{}.h5'.format(corruption_type))
    else:
        h5_name = os.path.join(DATA_DIR, 'modelnet_c', '{}_{}.h5'.format(corruption_type, level))
    f = h5py.File(h5_name, 'w')
    f.create_dataset('data', data=all_data)
    f.create_dataset('label', data=all_label)
    f.close()
    print("{} finished".format(h5_name))


def corrupt_data(all_data, type, level):
    if type == 'clean':
        return all_data
    corrupted_data = []
    for pcd in all_data:
        corrupted_pcd = corruptions[type](pcd, level)
        corrupted_data.append(corrupted_pcd)
    corrupted_data = np.stack(corrupted_data, axis=0)
    return corrupted_data


def main():
    all_data, all_label = load_data('test')
    for corruption_type in corruptions:
        for level in range(5):
            corrupted_data = corrupt_data(all_data, corruption_type, level)
            save_data(corrupted_data, all_label, corruption_type, level)
            if corruption_type == 'clean':
                break


if __name__ == '__main__':
    main()
