import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans, DBSCAN
from utils.util import BCE, PairEnum, Identity, AverageMeter, seed_torch, BCE_softlabels
from utils import ramps
from models.resnet import ResNet, BasicBlock
from data.cifarloader import CIFAR10Loader, CIFAR10LoaderMix, CIFAR100Loader, CIFAR100LoaderMix
from data.tinyimagenetloader import TinyImageNetLoader
from tqdm import tqdm
import numpy as np
import os
from models.NCL import NCLMemory
import wandb
from utils.fair_evals import cluster_acc

def fair_test(model, test_loader, args, cluster=True, ind=None, return_ind=False):
    model.eval()
    preds = np.array([])
    targets = np.array([])

    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(args.device), label.to(args.device)
        output1, output2 = model(x)
        if args.head == 'head1':
            output = torch.cat((output1, output2), dim=1)
        else:
            output = output2

        _, pred = output.max(1)
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())

    if cluster:
        if return_ind:
            acc, ind = cluster_acc(targets.astype(int), preds.astype(int), return_ind)
        else:
            acc = cluster_acc(targets.astype(int), preds.astype(int), return_ind)
        nmi, ari = nmi_score(targets, preds), ari_score(targets, preds)
        print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
    else:
        if ind is not None:
            ind = ind[:args.num_unlabeled_classes, :]
            idx = np.argsort(ind[:, 1])
            id_map = ind[idx, 0]
            id_map += args.num_labeled_classes

            targets_new = np.copy(targets)
            for i in range(args.num_unlabeled_classes):
                targets_new[targets == i + args.num_labeled_classes] = id_map[i]
            targets = targets_new

        preds = torch.from_numpy(preds)
        targets = torch.from_numpy(targets)
        correct = preds.eq(targets).float().sum(0)
        acc = float(correct / targets.size(0))
        print('Test acc {:.4f}'.format(acc))

    if return_ind:
        return acc, ind
    else:
        return acc

if __name__ == "__main__":
    import argparse
    parser = argparse.ArgumentParser(
            description='cluster',
            formatter_class=argparse.ArgumentDefaultsHelpFormatter)
    parser.add_argument('--lr', type=float, default=0.1)
    parser.add_argument('--gamma', type=float, default=0.1)
    parser.add_argument('--momentum', type=float, default=0.9)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--epochs', default=200, type=int)
    parser.add_argument('--rampup_length', default=150, type=int)
    parser.add_argument('--rampup_coefficient', type=float, default=50)
    parser.add_argument('--step_size', default=170, type=int)
    parser.add_argument('--batch_size', default=128, type=int)
    parser.add_argument('--num_unlabeled_classes', default=5, type=int)
    parser.add_argument('--num_labeled_classes', default=5, type=int)
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--warmup_model_dir', type=str, default='./data/experiments/pretrained/auto_novel/resnet_rotnet_cifar10.pth')
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--model_name', type=str, default='resnet')
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100, tinyimagenet')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--mode', type=str, default='train')
    parser.add_argument('--bce_type', type=str, default='cos')
    parser.add_argument('--hard_negative_start', default=1000, type=int)
    parser.add_argument('--knn', default=-1, type=int)
    parser.add_argument('--w_ncl_la', type=float, default=0.1)
    parser.add_argument('--w_ncl_ulb', type=float, default=1.0)
    parser.add_argument('--costhre', type=float, default=0.95)
    parser.add_argument('--m_size', default=2000, type=int)
    parser.add_argument('--m_t', type=float, default=0.05)
    parser.add_argument('--w_pos', type=float, default=0.2)
    parser.add_argument('--hard_iter', type=int, default=5)
    parser.add_argument('--num_hard', type=int, default=400)
    parser.add_argument('--wandb_mode', type=str, default='offline', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_entity', type=str, default='oatmealliu')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    device = torch.device("cuda" if args.cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    seed_torch(args.seed)

    runner_name = "ncl_all"
    model_dir = os.path.join(args.exp_root, runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+'{}.pth'.format(args.model_name)

    model = ResNet(BasicBlock, [2,2,2,2], args.num_labeled_classes, args.num_unlabeled_classes).to(device)

    num_classes = args.num_labeled_classes + args.num_unlabeled_classes

    if args.dataset_name == 'cifar10':
        mix_train_loader = CIFAR10LoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True, labeled_list=range(args.num_labeled_classes), unlabeled_list=range(args.num_labeled_classes, num_classes))
        unlabeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        unlabeled_eval_loader_test = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        labeled_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes))
        all_eval_loader = CIFAR10Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(num_classes))
    elif args.dataset_name == 'cifar100':
        mix_train_loader = CIFAR100LoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True, labeled_list=range(args.num_labeled_classes), unlabeled_list=range(args.num_labeled_classes, num_classes))
        unlabeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        unlabeled_eval_loader_test = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes, num_classes))
        labeled_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(args.num_labeled_classes))
        all_eval_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None, shuffle=False, target_list = range(num_classes))
    elif args.dataset_name == 'tinyimagenet':
        mix_train_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root, aug='twice', shuffle=True, class_list = range(args.num_labeled_classes, num_classes), subfolder='train')
        unlabeled_eval_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root, aug=None, shuffle=False, class_list = range(args.num_labeled_classes, num_classes), subfolder='train')
        unlabeled_eval_loader_test = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root, aug=None, shuffle=False, class_list = range(args.num_labeled_classes, num_classes), subfolder='val')
        labeled_eval_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root, aug=None, shuffle=False, class_list = range(args.num_labeled_classes), subfolder='val')
        all_eval_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root, aug=None, shuffle=False, class_list = range(num_classes), subfolder='val')


    model.load_state_dict(torch.load(args.model_dir))

    # =============================== Final Test ===============================
    print("="*100)
    print("NCL Results on {}".format(args.dataset_name))
    print("="*100)

    acc_list = []

    print('Head2: test on unlabeled classes')
    args.head = 'head2'
    _, ind = fair_test(model, unlabeled_eval_loader, args, return_ind=True)

    print('Evaluating on Head1')
    args.head = 'head1'

    print('test on labeled classes (test split)')
    acc = fair_test(model, labeled_eval_loader, args, cluster=False)
    acc_list.append(acc)

    print('test on unlabeled classes (test split)')
    acc = fair_test(model, unlabeled_eval_loader_test, args, cluster=False, ind=ind)
    acc_list.append(acc)

    print('test on all classes w/o clustering (test split)')
    acc = fair_test(model, all_eval_loader, args, cluster=False, ind=ind)
    acc_list.append(acc)

    print('test on all classes w/ clustering (test split)')
    acc = fair_test(model, all_eval_loader, args, cluster=True)
    acc_list.append(acc)

    print('Evaluating on Head2')
    args.head = 'head2'

    print('test on unlabeled classes (train split)')
    acc = fair_test(model, unlabeled_eval_loader, args)
    acc_list.append(acc)

    print('test on unlabeled classes (test split)')
    acc = fair_test(model, unlabeled_eval_loader_test, args)
    acc_list.append(acc)

    print('Acc List: Joint Head1->Old, New, All_wo_cluster, All_w_cluster, Head2->Train, Test')
    print(acc_list)
