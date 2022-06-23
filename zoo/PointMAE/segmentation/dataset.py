import numpy as np
import os
from torch.utils.data import Dataset
import torch
# from pointnet_util import pc_normalize
import json
import glob
import h5py



# class PartNormalDataset(Dataset):
#     def __init__(self, root='/mnt/lustre/share/ldkong/data/sets/ShapeNetPart', npoints=2500, split='train', class_choice=None, normal_channel=False):
#         self.npoints = npoints
#         self.root = root
#         # self.catfile = os.path.join(self.root, 'synsetoffset2category.txt')
#         self.catfile = '/mnt/lustre/share/ldkong/data/sets/ShapeNetPart/synsetoffset2category.txt'
#         self.cat = {}
#         self.normal_channel = normal_channel

#         with open(self.catfile, 'r') as f:
#             for line in f:
#                 ls = line.strip().split()
#                 self.cat[ls[0]] = ls[1]
#         self.cat = {k: v for k, v in self.cat.items()}
#         self.classes_original = dict(zip(self.cat, range(len(self.cat))))

#         if not class_choice is  None:
#             self.cat = {k:v for k,v in self.cat.items() if k in class_choice}
#         # print(self.cat)

#         self.meta = {}
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'), 'r') as f:
#             train_ids = set([str(d.split('/')[2]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'), 'r') as f:
#             val_ids = set([str(d.split('/')[2]) for d in json.load(f)])
#         with open(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'), 'r') as f:
#             test_ids = set([str(d.split('/')[2]) for d in json.load(f)])
#         for item in self.cat:
#             # print('category', item)
#             self.meta[item] = []
#             dir_point = os.path.join(self.root, self.cat[item])
#             fns = sorted(os.listdir(dir_point))
#             # print(fns[0][0:-4])
#             if split == 'trainval':
#                 fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
#             elif split == 'train':
#                 fns = [fn for fn in fns if fn[0:-4] in train_ids]
#             elif split == 'val':
#                 fns = [fn for fn in fns if fn[0:-4] in val_ids]
#             elif split == 'test':
#                 fns = [fn for fn in fns if fn[0:-4] in test_ids]
#             else:
#                 print('Unknown split: %s. Exiting..' % (split))
#                 exit(-1)

#             # print(os.path.basename(fns))
#             for fn in fns:
#                 token = (os.path.splitext(os.path.basename(fn))[0])
#                 self.meta[item].append(os.path.join(dir_point, token + '.txt'))

#         self.datapath = []
#         for item in self.cat:
#             for fn in self.meta[item]:
#                 self.datapath.append((item, fn))

#         self.classes = {}
#         for i in self.cat.keys():
#             self.classes[i] = self.classes_original[i]

#         # Mapping from category ('Chair') to a list of int [10,11,12,13] as segmentation labels
#         self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
#                             'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46],
#                             'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27],
#                             'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40],
#                             'Chair': [12, 13, 14, 15], 'Knife': [22, 23]}

#         # for cat in sorted(self.seg_classes.keys()):
#         #     print(cat, self.seg_classes[cat])

#         self.cache = {}  # from index to (point_set, cls, seg) tuple
#         self.cache_size = 20000


#     def __getitem__(self, index):
#         if index in self.cache:
#             point_set, cls, seg = self.cache[index]
#         else:
#             fn = self.datapath[index]
#             cat = self.datapath[index][0]
#             cls = self.classes[cat]
#             cls = np.array([cls]).astype(np.int32)
#             data = np.loadtxt(fn[1]).astype(np.float32)
#             if not self.normal_channel:
#                 point_set = data[:, 0:3]
#             else:
#                 point_set = data[:, 0:6]
#             seg = data[:, -1].astype(np.int32)
#             if len(self.cache) < self.cache_size:
#                 self.cache[index] = (point_set, cls, seg)
#         point_set[:, 0:3] = pc_normalize(point_set[:, 0:3])

#         choice = np.random.choice(len(seg), self.npoints, replace=True)
#         # resample
#         point_set = point_set[choice, :]
#         seg = seg[choice]

#         return point_set, cls, seg

#     def __len__(self):
#         return len(self.datapath)


class ShapeNetPart(Dataset):
    def __init__(self, num_points=2048, partition='train', class_choice=None, sub=None):
        self.data, self.label, self.seg = load_data_partseg(partition, sub)
        self.cat2id = {
            'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
            'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
            'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15
        }
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.num_points = num_points
        self.partition = partition        
        self.class_choice = class_choice

        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def __getitem__(self, item):
        pointcloud = self.data[item][:self.num_points]
        label = self.label[item]
        seg = self.seg[item][:self.num_points]  # part seg label
        if self.partition == 'trainval':
            pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]


class ShapeNetC(Dataset):
    def __init__(self, partition='train', class_choice=None, sub=None):
        self.data, self.label, self.seg = load_data_partseg(partition, sub)
        self.cat2id = {
            'airplane': 0, 'bag': 1, 'cap': 2, 'car': 3, 'chair': 4, 
            'earphone': 5, 'guitar': 6, 'knife': 7, 'lamp': 8, 'laptop': 9, 
            'motor': 10, 'mug': 11, 'pistol': 12, 'rocket': 13, 'skateboard': 14, 'table': 15
        }
        self.seg_num = [4, 2, 2, 4, 4, 3, 3, 2, 4, 2, 6, 2, 3, 3, 3, 3]  # number of parts for each category
        self.index_start = [0, 4, 6, 8, 12, 16, 19, 22, 24, 28, 30, 36, 38, 41, 44, 47]
        self.partition = partition        
        self.class_choice = class_choice

        if self.class_choice != None:
            id_choice = self.cat2id[self.class_choice]
            indices = (self.label == id_choice).squeeze()
            self.data = self.data[indices]
            self.label = self.label[indices]
            self.seg = self.seg[indices]
            self.seg_num_all = self.seg_num[id_choice]
            self.seg_start_index = self.index_start[id_choice]
        else:
            self.seg_num_all = 50
            self.seg_start_index = 0

    def __getitem__(self, item):
        pointcloud = self.data[item]
        label = self.label[item]
        seg = self.seg[item]  # part seg label
        if self.partition == 'trainval':
            pointcloud = translate_pointcloud(pointcloud)
            indices = list(range(pointcloud.shape[0]))
            np.random.shuffle(indices)
            pointcloud = pointcloud[indices]
            seg = seg[indices]
        return pointcloud, label, seg

    def __len__(self):
        return self.data.shape[0]



DATA_DIR = '/mnt/lustre/share/ldkong/data/sets/ShapeNetPart'
SHAPENET_C_DIR = '/mnt/lustre/share/jwren/to_kld/shapenet_c'

def load_data_partseg(partition, sub=None):
    all_data = []
    all_label = []
    all_seg = []
    if partition == 'trainval':
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', 'hdf5_data', '*train*.h5')) \
            + glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', 'hdf5_data', '*val*.h5'))
    elif partition == 'shapenet-c':
        file = os.path.join(SHAPENET_C_DIR, '%s.h5'%sub)
    else:
        file = glob.glob(os.path.join(DATA_DIR, 'shapenet_part_seg_hdf5_data', 'hdf5_data', '*%s*.h5'%partition))


    if partition == 'shapenet-c':
    # for h5_name in file:
        # f = h5py.File(h5_name, 'r+')
        f = h5py.File(file, 'r')
        data = f['data'][:].astype('float32')
        label = f['label'][:].astype('int64')
        seg = f['pid'][:].astype('int64')  # part seg label
        f.close()
        all_data.append(data)
        all_label.append(label)
        all_seg.append(seg)

    else:
        for h5_name in file:
            f = h5py.File(h5_name, 'r+')
            data = f['data'][:].astype('float32')
            label = f['label'][:].astype('int64')
            seg = f['pid'][:].astype('int64')
            f.close()
            all_data.append(data)
            all_label.append(label)
            all_seg.append(seg)


    all_data = np.concatenate(all_data, axis=0)
    all_label = np.concatenate(all_label, axis=0)
    all_seg = np.concatenate(all_seg, axis=0)
    return all_data, all_label, all_seg


def translate_pointcloud(pointcloud):
    xyz1 = np.random.uniform(low=2./3., high=3./2., size=[3])
    xyz2 = np.random.uniform(low=-0.2, high=0.2, size=[3])

    translated_pointcloud = np.add(np.multiply(pointcloud, xyz1), xyz2).astype('float32')
    return translated_pointcloud


def jitter_pointcloud(pointcloud, sigma=0.01, clip=0.02):
    N, C = pointcloud.shape
    pointcloud += np.clip(sigma * np.random.randn(N, C), -1*clip, clip)
    return pointcloud


def rotate_pointcloud(pointcloud):
    theta = np.pi*2 * np.random.uniform()
    rotation_matrix = np.array([[np.cos(theta), -np.sin(theta)],[np.sin(theta), np.cos(theta)]])
    pointcloud[:,[0,2]] = pointcloud[:,[0,2]].dot(rotation_matrix) # random rotation (x,z)
    return pointcloud


# if __name__ == '__main__':
#     data = ModelNetDataLoader('modelnet40_normal_resampled/', split='train', uniform=False, normal_channel=True)
#     DataLoader = torch.utils.data.DataLoader(data, batch_size=12, shuffle=True)
#     for point,label in DataLoader:
#         print(point.shape)
#         print(label.shape)