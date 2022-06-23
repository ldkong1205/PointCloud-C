#  Copyright (c) 2020. Author: Hanchen Wang, hw501@cam.ac.uk
#  Ref: https://github.com/yanx27/Pointnet_Pointnet2_pytorch/blob/master/data_utils/ShapeNetDataLoader.py
import os, json, torch, warnings, numpy as np
# from PC_Augmentation import pc_normalize
from torch.utils.data import Dataset
import glob
import h5py
warnings.filterwarnings('ignore')


# class PartNormalDataset(Dataset):
#     """
#     Data Source: https://shapenet.cs.stanford.edu/media/shapenetcore_partanno_segmentation_benchmark_v0_normal.zip
#     """
#     def __init__(self, root, num_point=2048, split='train', use_normal=False):
#         self.catfile = os.path.join(root, 'synsetoffset2category.txt')
#         self.use_normal = use_normal
#         self.num_point = num_point
#         self.cache_size = 20000
#         self.datapath = []
#         self.root = root
#         self.cache = {}
#         self.meta = {}
#         self.cat = {}

#         with open(self.catfile, 'r') as f:
#             for line in f:
#                 ls = line.strip().split()
#                 self.cat[ls[0]] = ls[1]
#         # self.cat -> {'class name': syn_id, ...}
#         # self.meta -> {'class name': file list, ...}
#         # self.classes -> {'class name': class id, ...}
#         # self.datapath -> [('class name', single file) , ...]
#         self.classes = dict(zip(self.cat, range(len(self.cat))))

#         train_ids = self.read_fns(os.path.join(self.root, 'train_test_split', 'shuffled_train_file_list.json'))
#         test_ids = self.read_fns(os.path.join(self.root, 'train_test_split', 'shuffled_test_file_list.json'))
#         val_ids = self.read_fns(os.path.join(self.root, 'train_test_split', 'shuffled_val_file_list.json'))
        
#         for item in self.cat:
#             dir_point = os.path.join(self.root, self.cat[item])
#             fns = sorted(os.listdir(dir_point))
#             self.meta[item] = []

#             if split is 'trainval':
#                 fns = [fn for fn in fns if ((fn[0:-4] in train_ids) or (fn[0:-4] in val_ids))]
#             elif split is 'test':
#                 fns = [fn for fn in fns if fn[0:-4] in test_ids]
#             else:
#                 print('Unknown split: %s [Option: ]. Exiting...' % split)
#                 exit(-1)

#             for fn in fns:
#                 token = (os.path.splitext(os.path.basename(fn))[0])
#                 self.meta[item].append(os.path.join(dir_point, token + '.txt'))

#         for item in self.cat:
#             for fn in self.meta[item]:
#                 self.datapath.append((item, fn))

#         self.seg_classes = {'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35],
#                             'Rocket': [41, 42, 43], 'Car': [8, 9, 10, 11], 'Laptop': [28, 29],
#                             'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Lamp': [24, 25, 26, 27],
#                             'Mug': [36, 37], 'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Knife': [22, 23],
#                             'Table': [47, 48, 49], 'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 
#                             'Chair': [12, 13, 14, 15]}

#     @staticmethod
#     def read_fns(path):
#         with open(path, 'r') as file:
#             ids = set([str(d.split('/')[2]) for d in json.load(file)])
#         return ids

#     def __getitem__(self, index):
#         if index in self.cache:
#             pts, cls, seg = self.cache[index]
#         else:
#             fn = self.datapath[index]
#             cat, pt = fn[0], np.loadtxt(fn[1]).astype(np.float32)
#             cls = np.array([self.classes[cat]]).astype(np.int32)
#             pts = pt[:, :6] if self.use_normal else pt[:, :3]
#             seg = pt[:, -1].astype(np.int32)
#             if len(self.cache) < self.cache_size:
#                 self.cache[index] = (pts, cls, seg)

#         choice = np.random.choice(len(seg), self.num_point, replace=True)
#         pts[:, 0:3] = pc_normalize(pts[:, 0:3])
#         pts, seg = pts[choice, :], seg[choice]

#         return pts, cls, seg

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




# if __name__ == "__main__":

#     root = '../data/shapenetcore_partanno_segmentation_benchmark_v0_normal/'
#     TRAIN_DATASET = PartNormalDataset(root=root, num_point=2048, split='trainval', use_normal=False)
#     trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=24, shuffle=True, num_workers=4)

#     for i, data in enumerate(trainDataLoader):
#         points, label, target = data

