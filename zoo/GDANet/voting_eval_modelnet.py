from __future__ import print_function
import os
import argparse
import torch
import torch.nn as nn
import torch.nn.functional as F
from util.data_util import ModelNet40
from model.GDANet_cls import GDANET
import numpy as np
from torch.utils.data import DataLoader
from util.util import cal_loss, IOStream
import sklearn.metrics as metrics


class PointcloudScale(object):
    def __init__(self, scale_low=2. / 3., scale_high=3. / 2., trans_low=-0.2, trans_high=0.2, trans_open=True):
        self.scale_low = scale_low
        self.scale_high = scale_high
        self.trans_low = trans_low
        self.trans_high = trans_high
        self.trans_open = trans_open  # whether add translation during voting or not

    def __call__(self, pc):
        bsize = pc.size()[0]
        for i in range(bsize):
            xyz1 = np.random.uniform(low=self.scale_low, high=self.scale_high, size=[3])
            xyz2 = np.random.uniform(low=self.trans_low, high=self.trans_high, size=[3])
            scales = torch.from_numpy(xyz1).float().cuda()
            trans = torch.from_numpy(xyz2).float().cuda() if self.trans_open else 0
            pc[i, :, 0:3] = torch.mul(pc[i, :, 0:3], scales)+trans
        return pc


def test(args, io):
    test_loader = DataLoader(ModelNet40(partition='test', num_points=args.num_points), num_workers=5,
                             batch_size=args.test_batch_size, shuffle=False, drop_last=False)

    device = torch.device("cuda" if args.cuda else "cpu")
    NUM_PEPEAT = 300
    NUM_VOTE = 10
    # Try to load models
    model = GDANET().to(device)
    model = nn.DataParallel(model)
    model.load_state_dict(torch.load(args.model_path))
    model = model.eval()
    best_acc = 0

    pointscale=PointcloudScale(scale_low=2. / 3., scale_high=3. / 2., trans_low=-0.2, trans_high=0.2, trans_open=True)
    for i in range(NUM_PEPEAT):
        test_true = []
        test_pred = []

        for data, label in test_loader:
            data, label = data.to(device), label.to(device).squeeze()
            pred = 0
            for v in range(NUM_VOTE):
                new_data = data
                batch_size = data.size()[0]
                if v > 0:
                    new_data.data = pointscale(new_data.data)
                with torch.no_grad():
                    pred += F.softmax(model(new_data.permute(0, 2, 1)), dim=1)
            pred /= NUM_VOTE
            label = label.view(-1)
            pred_choice = pred.max(dim=1)[1]
            test_true.append(label.cpu().numpy())
            test_pred.append(pred_choice.detach().cpu().numpy())
        test_true = np.concatenate(test_true)
        test_pred = np.concatenate(test_pred)
        test_acc = metrics.accuracy_score(test_true, test_pred)
        if test_acc > best_acc:
            best_acc = test_acc
        outstr = 'Voting %d, test acc: %.6f,' % (i, test_acc*100)
        io.cprint(outstr)

    final_outstr = 'Final voting result test acc: %.6f,' % (best_acc * 100)
    io.cprint(final_outstr)


def _init_():
    if not os.path.exists('checkpoints'):
        os.makedirs('checkpoints')
    if not os.path.exists('checkpoints/'+args.exp_name):
        os.makedirs('checkpoints/'+args.exp_name)

    os.system('cp voting_eval_modelnet.py checkpoints'+'/'+args.exp_name+'/'+'voting_eval_modelnet.py.backup')


if __name__ == "__main__":
    parser = argparse.ArgumentParser(description='3D Object Classification')
    parser.add_argument('--exp_name', type=str, default='GDANet', metavar='N',
                        help='Name of the experiment')
    parser.add_argument('--test_batch_size', type=int, default=16, metavar='batch_size',
                        help='Size of batch)')
    parser.add_argument('--no_cuda', type=bool, default=False,
                        help='enables CUDA training')
    parser.add_argument('--seed', type=int, default=1, metavar='S',
                        help='random seed (default: 1)')
    parser.add_argument('--num_points', type=int, default=1024,
                        help='num of points to use')
    parser.add_argument('--model_path', type=str, default='', metavar='N',
                        help='Pretrained model path')
    parser.add_argument('--trans_open', type=bool, default=True, metavar='N',
                        help='enables input translation during voting')
    args = parser.parse_args()

    _init_()

    io = IOStream('checkpoints/' + args.exp_name + '/%s_voting.log' % (args.exp_name))

    io.cprint(str(args))

    args.cuda = not args.no_cuda and torch.cuda.is_available()
    torch.manual_seed(args.seed)
    if args.cuda:
        io.cprint('Using GPU')
        torch.cuda.manual_seed(args.seed)
    else:
        io.cprint('Using CPU')

    test(args, io)
