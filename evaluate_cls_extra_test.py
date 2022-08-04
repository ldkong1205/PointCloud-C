import numpy as np
import h5py
import torch
import torch.nn as nn
from model import RPC
from tqdm import tqdm
from torch.utils.data import Dataset, DataLoader
import argparse


class ClsExtraTest(Dataset):
    def __init__(self, h5_path):
        f = h5py.File(h5_path)
        self.data = f['data'][:].astype('float32')
        f.close()

    def __getitem__(self, item):
        pointcloud = self.data[item]
        return pointcloud

    def __len__(self):
        return self.data.shape[0]


def test(args):
    args.cuda = not args.no_cuda and torch.cuda.is_available()
    device = torch.device("cuda" if args.cuda else "cpu")

    # load model
    model = RPC(args).to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path, map_location=device))
    model.eval()

    test_loader = DataLoader(
        ClsExtraTest(h5_path=args.h5_path),
        batch_size=args.test_batch_size,
        shuffle=False,
        drop_last=False
    )
    test_pred = []
    for pcd in tqdm(test_loader):
        pcd = pcd.to(device)
        pcd = pcd.permute(0, 2, 1)
        logits = model(pcd)
        preds = logits.argmax(dim=1)
        test_pred.append(preds.detach().cpu().numpy())

    test_pred = np.concatenate(test_pred)

    f = h5py.File(args.save_path, 'w')
    f.create_dataset('label', data=test_pred)
    f.close()


if __name__ == '__main__':
    parser = argparse.ArgumentParser(description='Example evaluation script for cls_extra_test_data')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--dropout', type=float, default=0.5,
                        help='dropout rate')
    parser.add_argument('--model_path', type=str, default='./RPC.t7', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--h5_path', type=str, default='./cls_extra_test_data.h5', metavar='N',
                        help='testset h5 path')
    parser.add_argument('--save_path', type=str, default='./results.h5', metavar='N',
                        help='results h5 path')

    args = parser.parse_args()

    test(args)
