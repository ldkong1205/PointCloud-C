'''
    ModelNet dataset. Support ModelNet40, XYZ channels. Up to 2048 points.
    Faster IO than ModelNetDataset in the first epoch.
'''

import os
import sys
import numpy as np
import h5py
BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'utils'))
import utils.provider_save as provider


# Download dataset for point cloud classification
DATA_DIR = os.path.join(ROOT_DIR, 'data')
if not os.path.exists(DATA_DIR):
    os.mkdir(DATA_DIR)
if not os.path.exists(os.path.join(DATA_DIR, 'modelnet40_ply_hdf5_2048')):
    www = 'https://shapenet.cs.stanford.edu/media/modelnet40_ply_hdf5_2048.zip'
    zipfile = os.path.basename(www)
    os.system('wget %s; unzip %s' % (www, zipfile))
    os.system('mv %s %s' % (zipfile[:-4], DATA_DIR))
    os.system('rm %s' % (zipfile))


def shuffle_data(data, labels):
    """ Shuffle data and labels.
        Input:
          data: B,N,... numpy array
          label: B,... numpy array
        Return:
          shuffled data, label and shuffle indices
    """
    idx = np.arange(len(labels))
    np.random.shuffle(idx)
    return data[idx, ...], labels[idx], idx

def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)


class ModelNetH5Dataset(object):
    def __init__(self, list_filename, batch_size = 32, npoints = 1024, shuffle=True):
        self.list_filename = list_filename
        self.batch_size = batch_size
        self.npoints = npoints
        self.shuffle = shuffle
        self.h5_files = getDataFiles(self.list_filename)
        self.reset()

    def reset(self):
        ''' reset order of h5 files '''
        self.file_idxs = np.arange(0, len(self.h5_files))
        if self.shuffle: np.random.shuffle(self.file_idxs)
        self.current_data = None
        self.current_label = None
        self.current_file_idx = 0
        self.batch_idx = 0
   
    def _augment_batch_data(self, batch_data, shuffle=False, jitter=False, rot=False, rdscale=False, shift=False):
        # rotated_data = provider.rotate_point_cloud(batch_data)
        # rotated_data = provider.rotate_perturbation_point_cloud(rotated_data)
        # jittered_data = provider.random_scale_point_cloud(rotated_data[:,:,0:3])
        # jittered_data = provider.shift_point_cloud(jittered_data)
        # jittered_data = provider.jitter_point_cloud(jittered_data)
        # rotated_data[:,:,0:3] = jittered_data
        # return provider.shuffle_points(rotated_data)
        if rot:
            batch_data = provider.rotate_point_cloud(batch_data)
            batch_data = provider.rotate_perturbation_point_cloud(batch_data)
        if rdscale:
            tmp_data = provider.random_scale_point_cloud(batch_data[:,:,0:3])
            batch_data[:,:,0:3] = tmp_data
        if shift:
            tmp_data = provider.shift_point_cloud(batch_data[:,:,0:3])
            batch_data[:,:,0:3] = tmp_data
        if jitter:
            tmp_data = provider.jitter_point_cloud(batch_data[:,:,0:3])
            batch_data[:,:,0:3] = tmp_data
        if shuffle:
            batch_data = provider.shuffle_points(batch_data)
        return batch_data

    def _rddrop_batch_data(self, data_batch):
        return provider.random_point_dropout(data_batch)

    def _rsmix_batch_data(self, batch_data, label_batch, beta=0.0, n_sample=512):
        return provider.rsmix_for_save(batch_data, label_batch, beta=beta, n_sample=512)
        
    def _get_data_filename(self):
        return self.h5_files[self.file_idxs[self.current_file_idx]]

    def _load_data_file(self, filename):
        self.current_data,self.current_label = load_h5(filename)
        self.current_label = np.squeeze(self.current_label)
        self.batch_idx = 0
        if self.shuffle:
            self.current_data, self.current_label, _ = shuffle_data(self.current_data,self.current_label)
    
    def _has_next_batch_in_file(self):
        return self.batch_idx*self.batch_size < self.current_data.shape[0]

    def num_channel(self):
        return 3

    def has_next_batch(self):
        # TODO: add backend thread to load data
        if (self.current_data is None) or (not self._has_next_batch_in_file()):
            if self.current_file_idx >= len(self.h5_files):
                return False
            self._load_data_file(self._get_data_filename())
            self.batch_idx = 0
            self.current_file_idx += 1
        return self._has_next_batch_in_file()

    def next_batch(self, augment=False, convda=False, rddrop=False, rsmix_prob=0.5, beta=0.0, 
                   n_sample=512, shuffle=False, jitter=False, rot=False, rdscale=False, shift=False):
        ''' returned dimension may be smaller than self.batch_size '''
        start_idx = self.batch_idx * self.batch_size
        end_idx = min((self.batch_idx+1) * self.batch_size, self.current_data.shape[0])
        bsize = end_idx - start_idx
        batch_label = np.zeros((bsize), dtype=np.int32)
        data_batch = self.current_data[start_idx:end_idx, 0:self.npoints, :].copy()
        label_batch = self.current_label[start_idx:end_idx].copy()
        self.batch_idx += 1
        # if augment: data_batch = self._augment_batch_data(data_batch) 
        '''
        revised from
        '''
        # TODO : Original data -> copy 로 바꾸기?
        data_original_batch = data_batch
        lam = np.zeros(data_batch.shape[0],dtype=float)
        label_batch_b = label_batch
        data_batch_a_mask = data_batch
        data_batch_b_mask = data_batch
        len_a_idx = np.ones(data_batch.shape[0],dtype=int)*1024
        len_b_idx = np.ones(data_batch.shape[0],dtype=int)*1024
        ###---KNN
        data_batch_2 = data_batch
        knn_data_batch_mixed = data_batch
        knn_lam = np.zeros(data_batch.shape[0],dtype=float)
        knn_data_batch_a_mask = data_batch
        knn_data_batch_b_mask = data_batch
        knn_len_a_idx = np.ones(data_batch.shape[0],dtype=int)*1024
        knn_len_b_idx = np.ones(data_batch.shape[0],dtype=int)*1024
        ###
        cut_rad = 0.0
        if augment: 
            r = np.random.rand(1)
            # r = 0.1 # for debug
            if convda: data_batch = self._augment_batch_data(data_batch, shuffle=shuffle, jitter=jitter, rot=rot, rdscale=rdscale, shift=shift)
            if rddrop: data_batch = self._rddrop_batch_data(data_batch)
            if beta > 0 and r < rsmix_prob:
                data_batch, lam, label_batch, label_batch_b, cut_rad, data_batch_a_mask, data_batch_b_mask, len_a_idx, len_b_idx, data_batch_2,\
                knn_data_batch_mixed, knn_lam, knn_data_batch_a_mask, knn_data_batch_b_mask, knn_len_a_idx, knn_len_b_idx = self._rsmix_batch_data(data_batch, label_batch, beta=beta, n_sample=512)
        '''
        to here
        '''
        return data_batch, label_batch, lam, label_batch_b, data_original_batch, cut_rad, data_batch_a_mask, data_batch_b_mask, len_a_idx, len_b_idx, data_batch_2,\
                knn_data_batch_mixed, knn_lam, knn_data_batch_a_mask, knn_data_batch_b_mask, knn_len_a_idx, knn_len_b_idx 

if __name__=='__main__':
    d = ModelNetH5Dataset('data/modelnet40_ply_hdf5_2048/train_files.txt')
    print(d.shuffle)
    print(d.has_next_batch())
    ps_batch, cls_batch = d.next_batch(True)
    print(ps_batch.shape)
    print(cls_batch.shape)
