import argparse
import os
import torch
import sys
import importlib
import numpy as np
from tqdm import tqdm
from dataset import ShapeNetC

from collections import defaultdict
from torch.autograd import Variable


BASE_DIR = os.path.dirname(os.path.abspath(__file__))
ROOT_DIR = BASE_DIR
sys.path.append(os.path.join(ROOT_DIR, 'models'))

seg_classes = {
    'Earphone': [16, 17, 18], 'Motorbike': [30, 31, 32, 33, 34, 35], 'Rocket': [41, 42, 43],
    'Car': [8, 9, 10, 11], 'Laptop': [28, 29], 'Cap': [6, 7], 'Skateboard': [44, 45, 46], 'Mug': [36, 37],
    'Guitar': [19, 20, 21], 'Bag': [4, 5], 'Lamp': [24, 25, 26, 27], 'Table': [47, 48, 49],
    'Airplane': [0, 1, 2, 3], 'Pistol': [38, 39, 40], 'Chair': [12, 13, 14, 15], 'Knife': [22, 23]
}
seg_label_to_cat = {}  # {0:Airplane, 1:Airplane, ...49:Table}
for cat in seg_classes.keys():
    for label in seg_classes[cat]:
        seg_label_to_cat[label] = cat

classes_str = ['aero','bag','cap','car','chair','ear','guitar','knife','lamp','lapt','moto','mug','Pistol','rock','stake','table']


def inplace_relu(m):
    classname = m.__class__.__name__
    if classname.find('ReLU') != -1:
        m.inplace=True

def to_categorical(y, num_classes):
    """ 1-hot encodes a tensor """
    new_y = torch.eye(num_classes)[y.cpu().data.numpy(),]
    if (y.is_cuda):
        return new_y.cuda()
    return new_y


def compute_overall_iou(pred, target, num_classes):
    shape_ious = []
    pred = pred.max(dim=2)[1]    # (batch_size, num_points)  the pred_class_idx of each point in each sample
    pred_np = pred.cpu().data.numpy()

    target_np = target.cpu().data.numpy()
    for shape_idx in range(pred.size(0)):   # sample_idx
        part_ious = []
        for part in range(num_classes):   # class_idx! no matter which category, only consider all part_classes of all categories, check all 50 classes
            # for target, each point has a class no matter which category owns this point! also 50 classes!!!
            # only return 1 when both belongs to this class, which means correct:
            I = np.sum(np.logical_and(pred_np[shape_idx] == part, target_np[shape_idx] == part))
            # always return 1 when either is belongs to this class:
            U = np.sum(np.logical_or(pred_np[shape_idx] == part, target_np[shape_idx] == part))

            F = np.sum(target_np[shape_idx] == part)

            if F != 0:
                iou = I / float(U)    #  iou across all points for this class
                part_ious.append(iou)   #  append the iou of this class
        shape_ious.append(np.mean(part_ious))   # each time append an average iou across all classes of this sample (sample_level!)
    return shape_ious   # [batch_size]


def parse_args():
    parser = argparse.ArgumentParser('Model')
    parser.add_argument('--model', type=str, default='pt', help='model name')
    parser.add_argument('--gpu', type=str, default='0', help='specify GPU devices')
    parser.add_argument('--ckpts', type=str, help='ckpts')
    return parser.parse_args()


def main(args):
    # def log_string(str):
    #     logger.info(str)
    #     print(str)

    '''HYPER PARAMETER'''
    os.environ["CUDA_VISIBLE_DEVICES"] = args.gpu

    '''CREATE DIR'''
    # timestr = str(datetime.datetime.now().strftime('%Y-%m-%d_%H-%M'))
    # exp_dir = Path('./log/')
    # exp_dir.mkdir(exist_ok=True)
    # exp_dir = exp_dir.joinpath('part_seg')
    # exp_dir.mkdir(exist_ok=True)
    # if args.log_dir is None:
    #     exp_dir = exp_dir.joinpath(timestr)
    # else:
    #     exp_dir = exp_dir.joinpath(args.log_dir)
    # exp_dir.mkdir(exist_ok=True)
    # checkpoints_dir = exp_dir.joinpath('checkpoints/')
    # checkpoints_dir.mkdir(exist_ok=True)
    # log_dir = exp_dir.joinpath('logs/')
    # log_dir.mkdir(exist_ok=True)

    '''LOG'''
    # args = parse_args()
    # logger = logging.getLogger("Model")
    # logger.setLevel(logging.INFO)
    # formatter = logging.Formatter('%(asctime)s - %(name)s - %(levelname)s - %(message)s')
    # file_handler = logging.FileHandler('%s/%s.txt' % (log_dir, args.model))
    # file_handler.setLevel(logging.INFO)
    # file_handler.setFormatter(formatter)
    # logger.addHandler(file_handler)
    # log_string('PARAMETER ...')
    # log_string(args)

    # root = args.root

    # TRAIN_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='trainval', normal_channel=args.normal)
    # TRAIN_DATASET = ShapeNetPart(partition='trainval', num_points=2048, class_choice=None)
    # trainDataLoader = torch.utils.data.DataLoader(TRAIN_DATASET, batch_size=args.batch_size, shuffle=True, num_workers=10, pin_memory=True, drop_last=True)
    
    # TEST_DATASET = PartNormalDataset(root=root, npoints=args.npoint, split='test', normal_channel=args.normal)
    # TEST_DATASET = ShapeNetPart(partition='test', num_points=2048, class_choice=None)
    TEST_DATASET = ShapeNetC(partition='shapenet-c', sub='add_local_4', class_choice=None)
    testDataLoader = torch.utils.data.DataLoader(TEST_DATASET, batch_size=16, shuffle=False, num_workers=10, pin_memory=True, drop_last=False)
    
    # log_string("The number of training data is: %d" % len(TRAIN_DATASET))
    print("The number of test data is: %d" % len(TEST_DATASET))

    num_classes = 16
    num_part = 50

    '''MODEL LOADING'''
    MODEL = importlib.import_module(args.model)
    # shutil.copy('models/%s.py' % args.model, str(exp_dir))
    # shutil.copy('models/pointnet2_utils.py', str(exp_dir))

    classifier = MODEL.get_model(num_part).cuda()
    classifier.apply(inplace_relu)
    print('# generator parameters:', sum(param.numel() for param in classifier.parameters()))

    if args.ckpts is not None:
        classifier.load_model_from_ckpt_test(args.ckpts)
        # classifier.load_state_dict(torch.load(args.ckpts))

## we use adamw and cosine scheduler
    # def add_weight_decay(model, weight_decay=1e-5, skip_list=()):
    #     decay = []
    #     no_decay = []
    #     for name, param in model.named_parameters():
    #         if not param.requires_grad:
    #             continue  # frozen weights
    #         if len(param.shape) == 1 or name.endswith(".bias") or 'token' in name or name in skip_list:
    #                     # print(name)
    #             no_decay.append(param)
    #         else:
    #             decay.append(param)
    #     return [
    #                 {'params': no_decay, 'weight_decay': 0.},
    #                 {'params': decay, 'weight_decay': weight_decay}]

    # param_groups = add_weight_decay(classifier, weight_decay=0.05)
    # optimizer = optim.AdamW(param_groups, lr= args.learning_rate, weight_decay=0.05 )

    # scheduler = CosineLRScheduler(
    #     optimizer,
    #     t_initial=args.epoch,
    #     t_mul=1,
    #     lr_min=1e-6,
    #     decay_rate=0.1,
    #     warmup_lr_init=1e-6,
    #     warmup_t=args.warmup_epoch,
    #     cycle_limit=1,
    #     t_in_epochs=True
    # )

    # best_acc = 0
    # global_epoch = 0
    # best_class_avg_iou = 0
    # best_inctance_avg_iou = 0

    # classifier.zero_grad()

    metrics = defaultdict(lambda: list())
    hist_acc = []
    shape_ious = []
    total_per_cat_iou = np.zeros((16)).astype(np.float32)
    total_per_cat_seen = np.zeros((16)).astype(np.int32)

    for batch_id, (points, label, target) in tqdm(enumerate(testDataLoader), total=len(testDataLoader), smoothing=0.9):
        batch_size, num_point, _ = points.size()
        points, label, target = Variable(points.float()), Variable(label.long()), Variable(target.long())
        points = points.transpose(2, 1)
        points, label, target = points.cuda(non_blocking=True), label.squeeze().cuda(non_blocking=True), target.cuda(non_blocking=True)

        with torch.no_grad():
            seg_pred = classifier(points, to_categorical(label, num_classes))  # b,n,50

        # instance iou without considering the class average at each batch_size:
        batch_shapeious = compute_overall_iou(seg_pred, target, num_part)  # [b]
        shape_ious += batch_shapeious  # iou +=, equals to .append

        # per category iou at each batch_size:
        for shape_idx in range(seg_pred.size(0)):  # sample_idx
            cur_gt_label = label[shape_idx]  # label[sample_idx]
            total_per_cat_iou[cur_gt_label] += batch_shapeious[shape_idx]
            total_per_cat_seen[cur_gt_label] += 1

        # accuracy:
        seg_pred = seg_pred.contiguous().view(-1, num_part)
        target = target.view(-1, 1)[:, 0]
        pred_choice = seg_pred.data.max(1)[1]
        correct = pred_choice.eq(target.data).cpu().sum()
        metrics['accuracy'].append(correct.item() / (batch_size * num_point))

    hist_acc += metrics['accuracy']
    metrics['accuracy'] = np.mean(hist_acc)
    metrics['shape_avg_iou'] = np.mean(shape_ious)
    for cat_idx in range(16):
        if total_per_cat_seen[cat_idx] > 0:
            total_per_cat_iou[cat_idx] = total_per_cat_iou[cat_idx] / total_per_cat_seen[cat_idx]

    # First we need to calculate the iou of each class and the avg class iou:
    class_iou = 0
    for cat_idx in range(16):
        class_iou += total_per_cat_iou[cat_idx]
        print(classes_str[cat_idx] + ' iou: ' + str(total_per_cat_iou[cat_idx]))  # print the iou of each class
    avg_class_iou = class_iou / 16
    outstr = 'Test :: test acc: %f  test class mIOU: %f, test instance mIOU: %f' % (metrics['accuracy'], avg_class_iou, metrics['shape_avg_iou'])
    print(outstr)



if __name__ == '__main__':
    args = parse_args()
    main(args)