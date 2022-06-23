'''
RSMix:
@Author: Dogyoon Lee
@Contact: dogyoonlee@gmail.com
@File: provider.py
@Time: 2020/11/23 13:46 PM
'''


import os
import sys
import numpy as np
import h5py
# import tensorflow as tf
import random

BASE_DIR = os.path.dirname(os.path.abspath(__file__))
sys.path.append(BASE_DIR)

# def set_random_seed(seed=1):
#     # set random_seed
#     random.seed(seed)
#     np.random.seed(seed)
#     tf.set_random_seed(seed)

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

def shuffle_points(batch_data):
    """ Shuffle orders of points in each point cloud -- changes FPS behavior.
        Use the same shuffling idx for the entire batch.
        Input:
            BxNxC array
        Output:
            BxNxC array
    """
    idx = np.arange(batch_data.shape[1])
    np.random.shuffle(idx)
    return batch_data[:,idx,:]

def rotate_point_cloud(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud_z(batch_data):
    """ Randomly rotate the point clouds to augument the dataset
        rotation is per shape based along up direction
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, sinval, 0],
                                    [-sinval, cosval, 0],
                                    [0, 0, 1]])
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud_with_normal(batch_xyz_normal):
    ''' Randomly rotate XYZ, normal point cloud.
        Input:
            batch_xyz_normal: B,N,6, first three channels are XYZ, last 3 all normal
        Output:
            B,N,6, rotated XYZ, normal point cloud
    '''
    for k in range(batch_xyz_normal.shape[0]):
        rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_xyz_normal[k,:,0:3]
        shape_normal = batch_xyz_normal[k,:,3:6]
        batch_xyz_normal[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        batch_xyz_normal[k,:,3:6] = np.dot(shape_normal.reshape((-1, 3)), rotation_matrix)
    return batch_xyz_normal

def rotate_perturbation_point_cloud_with_normal(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx6 array, original batch of point clouds and point normals
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in list(range(batch_data.shape[0])):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k,:,0:3]
        shape_normal = batch_data[k,:,3:6]
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), R)
        rotated_data[k,:,3:6] = np.dot(shape_normal.reshape((-1, 3)), R)
    return rotated_data


def rotate_point_cloud_by_angle(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in list(range(batch_data.shape[0])):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k,:,0:3]
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
    return rotated_data

def rotate_point_cloud_by_angle_with_normal(batch_data, rotation_angle):
    """ Rotate the point cloud along up direction with certain angle.
        Input:
          BxNx6 array, original batch of point clouds with normal
          scalar, angle of rotation
        Return:
          BxNx6 array, rotated batch of point clouds iwth normal
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        #rotation_angle = np.random.uniform() * 2 * np.pi
        cosval = np.cos(rotation_angle)
        sinval = np.sin(rotation_angle)
        rotation_matrix = np.array([[cosval, 0, sinval],
                                    [0, 1, 0],
                                    [-sinval, 0, cosval]])
        shape_pc = batch_data[k,:,0:3]
        shape_normal = batch_data[k,:,3:6]
        rotated_data[k,:,0:3] = np.dot(shape_pc.reshape((-1, 3)), rotation_matrix)
        rotated_data[k,:,3:6] = np.dot(shape_normal.reshape((-1,3)), rotation_matrix)
    return rotated_data



def rotate_perturbation_point_cloud(batch_data, angle_sigma=0.06, angle_clip=0.18):
    """ Randomly perturb the point clouds by small rotations
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, rotated batch of point clouds
    """
    rotated_data = np.zeros(batch_data.shape, dtype=np.float32)
    for k in range(batch_data.shape[0]):
        angles = np.clip(angle_sigma*np.random.randn(3), -angle_clip, angle_clip)
        Rx = np.array([[1,0,0],
                       [0,np.cos(angles[0]),-np.sin(angles[0])],
                       [0,np.sin(angles[0]),np.cos(angles[0])]])
        Ry = np.array([[np.cos(angles[1]),0,np.sin(angles[1])],
                       [0,1,0],
                       [-np.sin(angles[1]),0,np.cos(angles[1])]])
        Rz = np.array([[np.cos(angles[2]),-np.sin(angles[2]),0],
                       [np.sin(angles[2]),np.cos(angles[2]),0],
                       [0,0,1]])
        R = np.dot(Rz, np.dot(Ry,Rx))
        shape_pc = batch_data[k, ...]
        rotated_data[k, ...] = np.dot(shape_pc.reshape((-1, 3)), R)
    return rotated_data


# def jitter_point_cloud(batch_data, sigma=0.01, clip=0.05):
def jitter_point_cloud(batch_data, sigma=0.01, clip=0.02):
    """ Randomly jitter points. jittering is per point.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, jittered batch of point clouds
    """
    B, N, C = batch_data.shape
    assert(clip > 0)
    jittered_data = np.clip(sigma * np.random.randn(B, N, C), -1*clip, clip)
    jittered_data += batch_data
    return jittered_data

# def shift_point_cloud(batch_data, shift_range=0.1):
def shift_point_cloud(batch_data, shift_range=0.2):
    """ Randomly shift point cloud. Shift is per point cloud.
        Input:
          BxNx3 array, original batch of point clouds
        Return:
          BxNx3 array, shifted batch of point clouds
    """
    B, N, C = batch_data.shape
    shifts = np.random.uniform(-shift_range, shift_range, (B,3))
    for batch_index in range(B):
        batch_data[batch_index,:,:] += shifts[batch_index,:]
    return batch_data


# def random_scale_point_cloud(batch_data, scale_low=0.8, scale_high=1.25):
def random_scale_point_cloud(batch_data, scale_low=2./3., scale_high=3./2.):
    """ Randomly scale the point cloud. Scale is per point cloud.
        Input:
            BxNx3 array, original batch of point clouds
        Return:
            BxNx3 array, scaled batch of point clouds
    """
    B, N, C = batch_data.shape
    scales = np.random.uniform(scale_low, scale_high, B)
    for batch_index in range(B):
        batch_data[batch_index,:,:] *= scales[batch_index]
    return batch_data

def random_point_dropout(batch_pc, max_dropout_ratio=0.875):
    ''' batch_pc: BxNx3 '''
    for b in range(batch_pc.shape[0]):
        dropout_ratio =  np.random.random()*max_dropout_ratio # 0~0.875
        drop_idx = np.where(np.random.random((batch_pc.shape[1]))<=dropout_ratio)[0]
        if len(drop_idx)>0:
            batch_pc[b,drop_idx,:] = batch_pc[b,0,:] # set to the first point
    return batch_pc


def getDataFiles(list_filename):
    return [line.rstrip() for line in open(list_filename)]

def load_h5(h5_filename):
    f = h5py.File(h5_filename)
    data = f['data'][:]
    label = f['label'][:]
    return (data, label)

def loadDataFile(filename):
    return load_h5(filename)


# for rsmix @@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@@
def knn_points(k, xyz, query, nsample=512):
    B, N, C = xyz.shape
    _, S, _ = query.shape # S=1
    
    tmp_idx = np.arange(N)
    group_idx = np.repeat(tmp_idx[np.newaxis,np.newaxis,:], B, axis=0)
    sqrdists = square_distance(query, xyz) # Bx1,N #제곱거리
    tmp = np.sort(sqrdists, axis=2)
    knn_dist = np.zeros((B,1))
    for i in range(B):
        knn_dist[i][0] = tmp[i][0][k]
        group_idx[i][sqrdists[i]>knn_dist[i][0]]=N
    # group_idx[sqrdists > radius ** 2] = N
    # print("group idx : \n",group_idx)
    # group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample] # for torch.tensor
    group_idx = np.sort(group_idx, axis=2)[:, :, :nsample]
    # group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    tmp_idx = group_idx[:,:,0]
    group_first = np.repeat(tmp_idx[:,np.newaxis,:], nsample, axis=2)
    # repeat the first value of the idx in each batch 
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx
    
def cut_points_knn(data_batch, idx, radius, nsample=512, k=512):
    """
        input
        points : BxNx3(=6 with normal)
        idx : Bx1 one scalar(int) between 0~len(points)
        
        output
        idx : Bxn_sample
    """
    B, N, C = data_batch.shape
    B, S = idx.shape
    query_points = np.zeros((B,1,C))
    # print("idx : \n",idx)
    for i in range(B):
        query_points[i][0]=data_batch[i][idx[i][0]] # Bx1x3(=6 with normal)
    # B x n_sample
    group_idx = knn_points(k=k, xyz=data_batch[:,:,:3], query=query_points[:,:,:3], nsample=nsample)
    return group_idx, query_points # group_idx: 16x?x6, query_points: 16x1x6

def cut_points(data_batch, idx, radius, nsample=512):
    """
        input
        points : BxNx3(=6 with normal)
        idx : Bx1 one scalar(int) between 0~len(points)
        
        output
        idx : Bxn_sample
    """
    B, N, C = data_batch.shape
    B, S = idx.shape
    query_points = np.zeros((B,1,C))
    # print("idx : \n",idx)
    for i in range(B):
        query_points[i][0]=data_batch[i][idx[i][0]] # Bx1x3(=6 with normal)
    # B x n_sample
    group_idx = query_ball_point_for_rsmix(radius, nsample, data_batch[:,:,:3], query_points[:,:,:3])
    return group_idx, query_points # group_idx: 16x?x6, query_points: 16x1x6


def query_ball_point_for_rsmix(radius, nsample, xyz, new_xyz):
    """
    Input:
        radius: local region radius
        nsample: max sample number in local region
        xyz: all points, [B, N, 3]
        new_xyz: query points, [B, S, 3]
    Return:
        group_idx: grouped points index, [B, S, nsample], S=1
    """
    # device = xyz.device
    B, N, C = xyz.shape
    _, S, _ = new_xyz.shape
    # group_idx = torch.arange(N, dtype=torch.long).to(device).view(1, 1, N).repeat([B, S, 1])
    tmp_idx = np.arange(N)
    group_idx = np.repeat(tmp_idx[np.newaxis,np.newaxis,:], B, axis=0)
    
    sqrdists = square_distance(new_xyz, xyz)
    group_idx[sqrdists > radius ** 2] = N
    
    # group_idx = group_idx.sort(dim=-1)[0][:, :, :nsample] # for torch.tensor
    group_idx = np.sort(group_idx, axis=2)[:, :, :nsample]
    # group_first = group_idx[:, :, 0].view(B, S, 1).repeat([1, 1, nsample])
    tmp_idx = group_idx[:,:,0]
    group_first = np.repeat(tmp_idx[:,np.newaxis,:], nsample, axis=2)
    # repeat the first value of the idx in each batch 
    mask = group_idx == N
    group_idx[mask] = group_first[mask]
    return group_idx

def square_distance(src, dst):
    """
    Calculate Euclid distance between each two points.

    src^T * dst = xn * xm + yn * ym + zn * zm；
    sum(src^2, dim=-1) = xn*xn + yn*yn + zn*zn;
    sum(dst^2, dim=-1) = xm*xm + ym*ym + zm*zm;
    dist = (xn - xm)^2 + (yn - ym)^2 + (zn - zm)^2
         = sum(src**2,dim=-1)+sum(dst**2,dim=-1)-2*src^T*dst

    Input:
        src: source points, [B, N, C]
        dst: target points, [B, M, C]
    Output:
        dist: per-point square distance, [B, N, M]
    """
    B, N, _ = src.shape
    _, M, _ = dst.shape
    # dist = -2 * torch.matmul(src, dst.permute(0, 2, 1))
    # dist += torch.sum(src ** 2, -1).view(B, N, 1)
    # dist += torch.sum(dst ** 2, -1).view(B, 1, M)

    dist = -2 * np.matmul(src, dst.transpose(0, 2, 1))
    dist += np.sum(src ** 2, -1).reshape(B, N, 1)
    dist += np.sum(dst ** 2, -1).reshape(B, 1, M)
    
    return dist


def pts_num_ctrl(pts_erase_idx, pts_add_idx):
    '''
        input : pts - to erase 
                pts - to add
        output :pts - to add (number controled)
    '''
    if len(pts_erase_idx)>=len(pts_add_idx):
        num_diff = len(pts_erase_idx)-len(pts_add_idx)
        if num_diff == 0:
            pts_add_idx_ctrled = pts_add_idx
        else:
            pts_add_idx_ctrled = np.append(pts_add_idx, pts_add_idx[np.random.randint(0, len(pts_add_idx), size=num_diff)])
    else:
        pts_add_idx_ctrled = np.sort(np.random.choice(pts_add_idx, size=len(pts_erase_idx), replace=False))
    return pts_add_idx_ctrled

def rsmix(data_batch, label_batch, beta=1.0, n_sample=512, KNN=False):
    cut_rad = np.random.beta(beta, beta)
    rand_index = np.random.choice(data_batch.shape[0],data_batch.shape[0], replace=False) # label dim : (16,) for model
    
    if len(label_batch.shape) is 1:
        label_batch = np.expand_dims(label_batch, axis=1)
        
    label_a = label_batch[:,0]
    label_b = label_batch[rand_index][:,0]
        
    data_batch_rand = data_batch[rand_index] # BxNx3(with normal=6)
    rand_idx_1 = np.random.randint(0,data_batch.shape[1], (data_batch.shape[0],1))
    rand_idx_2 = np.random.randint(0,data_batch.shape[1], (data_batch.shape[0],1))
    if KNN:
        knn_para = min(int(np.ceil(cut_rad*n_sample)),n_sample)
        pts_erase_idx, query_point_1 = cut_points_knn(data_batch, rand_idx_1, cut_rad, nsample=n_sample, k=knn_para) # B x num_points_in_radius_1 x 3(or 6)
        pts_add_idx, query_point_2 = cut_points_knn(data_batch_rand, rand_idx_2, cut_rad, nsample=n_sample, k=knn_para) # B x num_points_in_radius_2 x 3(or 6)
    else:
        pts_erase_idx, query_point_1 = cut_points(data_batch, rand_idx_1, cut_rad, nsample=n_sample) # B x num_points_in_radius_1 x 3(or 6)
        pts_add_idx, query_point_2 = cut_points(data_batch_rand, rand_idx_2, cut_rad, nsample=n_sample) # B x num_points_in_radius_2 x 3(or 6)
    
    query_dist = query_point_1[:,:,:3] - query_point_2[:,:,:3]
    
    pts_replaced = np.zeros((1,data_batch.shape[1],data_batch.shape[2]))
    lam = np.zeros(data_batch.shape[0],dtype=float)

    for i in range(data_batch.shape[0]):
        if pts_erase_idx[i][0][0]==data_batch.shape[1]:
            tmp_pts_replaced = np.expand_dims(data_batch[i], axis=0)
            lam_tmp = 0
        elif pts_add_idx[i][0][0]==data_batch.shape[1]:
            pts_erase_idx_tmp = np.unique(pts_erase_idx[i].reshape(n_sample,),axis=0)
            tmp_pts_erased = np.delete(data_batch[i], pts_erase_idx_tmp, axis=0) # B x N-num_rad_1 x 3(or 6)
            dup_points_idx = np.random.randint(0,len(tmp_pts_erased), size=len(pts_erase_idx_tmp))
            tmp_pts_replaced = np.expand_dims(np.concatenate((tmp_pts_erased, data_batch[i][dup_points_idx]), axis=0), axis=0)
            lam_tmp = 0
        else:
            pts_erase_idx_tmp = np.unique(pts_erase_idx[i].reshape(n_sample,),axis=0)
            pts_add_idx_tmp = np.unique(pts_add_idx[i].reshape(n_sample,),axis=0)
            pts_add_idx_ctrled_tmp = pts_num_ctrl(pts_erase_idx_tmp,pts_add_idx_tmp)
            tmp_pts_erased = np.delete(data_batch[i], pts_erase_idx_tmp, axis=0) # B x N-num_rad_1 x 3(or 6)
            # input("INPUT : ")
            tmp_pts_to_add = np.take(data_batch_rand[i], pts_add_idx_ctrled_tmp, axis=0)
            tmp_pts_to_add[:,:3] = query_dist[i]+tmp_pts_to_add[:,:3]
            
            tmp_pts_replaced = np.expand_dims(np.vstack((tmp_pts_erased,tmp_pts_to_add)), axis=0)
            
            lam_tmp = len(pts_add_idx_ctrled_tmp)/(len(pts_add_idx_ctrled_tmp)+len(tmp_pts_erased))
        
        pts_replaced = np.concatenate((pts_replaced, tmp_pts_replaced),axis=0)
        lam[i] = lam_tmp
    
    data_batch_mixed = np.delete(pts_replaced, [0], axis=0)    
        
        
    return data_batch_mixed, lam, label_a, label_b

