from util.data_util import PartNormalDataset, ShapeNetC
from torch.utils.data import DataLoader
from torch.autograd import Variable



# test_data = PartNormalDataset(npoints=2048, split='test', normalize=False)
test_data = ShapeNetC(partition='shapenet-c', sub='clean', class_choice=None)
print("The number of test data is: {}".format(len(test_data)))

test_loader = DataLoader(test_data, batch_size=16, shuffle=False, num_workers=6, drop_last=False)


for batch_id, (points, label, target, norm_plt) in enumerate(test_loader):
        batch_size, num_point, _ = points.size()
        points, label, target, norm_plt = Variable(points.float()), Variable(label.long()), Variable(target.long()), Variable(norm_plt.float())
        points = points.transpose(2, 1)
        norm_plt = norm_plt.transpose(2, 1)
        points, label, target, norm_plt = points.cuda(non_blocking=True), label.squeeze().cuda(non_blocking=True), target.cuda(non_blocking=True), norm_plt.cuda(non_blocking=True)
