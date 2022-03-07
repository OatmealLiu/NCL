import torch
import torch.nn as nn
import torch.nn.functional as F
from torch.optim import SGD, lr_scheduler
from sklearn.metrics.cluster import normalized_mutual_info_score as nmi_score
from sklearn.metrics import adjusted_rand_score as ari_score
from sklearn.cluster import KMeans, DBSCAN
from utils.util import BCE, PairEnum, Identity, AverageMeter, seed_torch, BCE_softlabels
from utils import ramps
from models.resnet import ResNet, BasicBlock, ResNetTri
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
        if args.step == 'first':
            output1, output2 = model(x)
            if args.head == 'head1':
                output = torch.cat((output1, output2), dim=1)
            else:
                output = output2
        elif args.step == 'second' and args.test_new == 'new1' and args.head != 'head1':
            output1, output2 = model(x)
            if args.head == 'head1':
                output = torch.cat((output1, output2), dim=1)
            else:
                output = output2
        else:
            output1, output2, output3 = model(x)
            if args.head == 'head1':
                output = torch.cat((output1, output2, output3), dim=1)
            elif args.head == 'head2':
                output = output2
            elif args.head == 'head3':
                output = output3

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
            if args.step == 'first' or args.test_new == 'new1':
                ind = ind[:args.num_unlabeled_classes1, :]
                idx = np.argsort(ind[:, 1])
                id_map = ind[idx, 0]
                id_map += args.num_labeled_classes

                # targets_new = targets
                targets_new = np.copy(targets)
                for i in range(args.num_unlabeled_classes1):
                    targets_new[targets == i + args.num_labeled_classes] = id_map[i]
                targets = targets_new
            else:
                ind = ind[:args.num_unlabeled_classes2, :]
                idx = np.argsort(ind[:, 1])
                id_map = ind[idx, 0]
                id_map += args.num_labeled_classes+args.num_unlabeled_classes1

                # targets_new = targets
                targets_new = np.copy(targets)
                for i in range(args.num_unlabeled_classes2):
                    targets_new[targets == i + args.num_labeled_classes+args.num_unlabeled_classes1] = id_map[i]
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
    parser.add_argument('--num_unlabeled_classes1', default=10, type=int)
    parser.add_argument('--num_unlabeled_classes2', default=10, type=int)
    parser.add_argument('--num_labeled_classes', default=80, type=int)
    parser.add_argument('--dataset_root', type=str, default='./data/datasets/CIFAR/')
    parser.add_argument('--exp_root', type=str, default='./data/experiments/')
    parser.add_argument('--warmup_model_dir', type=str, default='./data/experiments/pretrained/auto_novel/resnet_rotnet_cifar10.pth')
    parser.add_argument('--topk', default=5, type=int)
    parser.add_argument('--model_name', type=str, default='resnet')
    parser.add_argument('--dataset_name', type=str, default='cifar10', help='options: cifar10, cifar100, tinyimagenet')
    parser.add_argument('--seed', default=0, type=int)
    parser.add_argument('--mode', type=str, default='eval')
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
    parser.add_argument('--step', type=str, default='first', choices=['first', 'second'])
    parser.add_argument('--first_step_dir', type=str,
                        default='./results/two_incd_cifar100_DTC/DTC_cifar100_incd_resnet18_80.pth')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    device = torch.device("cuda" if args.cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    seed_torch(args.seed)

    runner_name = "two_NCL_incd_train_tinyimagenet"

    model_dir = os.path.join(args.exp_root, runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+ args.step + '_'+'{}.pth'.format(args.model_name)

    if args.mode == 'eval' and args.step == 'first':
        num_classes = args.num_labeled_classes + args.num_unlabeled_classes1
        mix_train_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root,
                                              aug='twice', shuffle=True,
                                              class_list=range(args.num_labeled_classes, num_classes), subfolder='train')
        unlabeled_val_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root,
                                                  aug=None, shuffle=False,
                                                  class_list=range(args.num_labeled_classes, num_classes),
                                                  subfolder='train')
        unlabeled_test_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root,
                                                   aug=None, shuffle=False,
                                                   class_list=range(args.num_labeled_classes, num_classes),
                                                   subfolder='val')
        labeled_test_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root,
                                                 aug=None, shuffle=False, class_list=range(args.num_labeled_classes),
                                                 subfolder='val')
        all_test_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root,
                                             aug=None,
                                             shuffle=False, class_list=range(num_classes), subfolder='val')


        model = ResNet(BasicBlock, [2, 2, 2, 2], args.num_labeled_classes,
                       args.num_unlabeled_classes1+args.num_unlabeled_classes2).to(device)
        model.head2 = nn.Linear(512, args.num_unlabeled_classes1).to(device)
        state_dict = torch.load(args.model_dir)
        model.load_state_dict(state_dict, strict=False)


        # LOOK: OUR TEST FLOW
        # =============================== Final Test ===============================
        print("=" * 150)
        print("\t\t\t\tFirst: test function 1")
        print("=" * 150)
        acc_list = []

        print('Head2: test on unlabeled classes')
        args.head = 'head2'
        _, ind = fair_test(model, unlabeled_val_loader, args, return_ind=True)

        print('Evaluating on Head1')
        args.head = 'head1'

        print('test on labeled classes (test split)')
        acc = fair_test(model, labeled_test_loader, args, cluster=False)
        acc_list.append(acc)

        print('test on unlabeled classes (test split)')
        acc = fair_test(model, unlabeled_test_loader, args, cluster=False, ind=ind)
        acc_list.append(acc)

        print('test on all classes w/o clustering (test split)')
        acc = fair_test(model, all_test_loader, args, cluster=False, ind=ind)
        acc_list.append(acc)

        print('test on all classes w/ clustering (test split)')
        acc = fair_test(model, all_test_loader, args, cluster=True)
        acc_list.append(acc)

        print('Evaluating on Head2')
        args.head = 'head2'

        print('test on unlabeled classes (train split)')
        acc = fair_test(model, unlabeled_val_loader, args)
        acc_list.append(acc)

        print('test on unlabeled classes (test split)')
        acc = fair_test(model, unlabeled_test_loader, args)
        acc_list.append(acc)

        print('Acc List: Synthesized Head1->Old, New, All_wo_cluster, All_w_cluster, Head2->Train, Test')
        print(acc_list)
    elif args.mode == 'eval' and args.step == 'second':
        num_classes = args.num_labeled_classes + args.num_unlabeled_classes1 + args.num_unlabeled_classes2
        mix_train_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root,
                                              aug='twice', shuffle=True,
                                              class_list=range(args.num_labeled_classes+args.num_unlabeled_classes1,
                                                                     num_classes), subfolder='train')
        unlabeled_val_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root,
                                                  aug=None, shuffle=False,
                                                  class_list=range(args.num_labeled_classes+args.num_unlabeled_classes1,
                                                                   num_classes),
                                                  subfolder='train')
        unlabeled_test_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root,
                                                   aug=None, shuffle=False,
                                                   class_list=range(args.num_labeled_classes+args.num_unlabeled_classes1,
                                                                    num_classes),
                                                   subfolder='val')
        labeled_test_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root,
                                                 aug=None, shuffle=False, class_list=range(args.num_labeled_classes),
                                                 subfolder='val')
        all_test_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root,
                                             aug=None,
                                             shuffle=False, class_list=range(num_classes), subfolder='val')

        # Previous step Novel classes dataloader
        p_unlabeled_val_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root,
                                                  aug=None, shuffle=False,
                                                  class_list=range(args.num_labeled_classes,
                                                                args.num_labeled_classes + args.num_unlabeled_classes1),
                                                  subfolder='train')
        p_unlabeled_test_loader = TinyImageNetLoader(batch_size=args.batch_size, num_workers=8, path=args.dataset_root,
                                                   aug=None, shuffle=False,
                                                   class_list=range(args.num_labeled_classes,
                                                                 args.num_labeled_classes + args.num_unlabeled_classes1),
                                                   subfolder='val')

        # Old Model
        old_model = ResNet(BasicBlock, [2, 2, 2, 2], args.num_labeled_classes,
                       args.num_unlabeled_classes1 + args.num_unlabeled_classes2).to(device)
        old_model.head2 = nn.Linear(512, args.num_unlabeled_classes1).to(device)
        old_state_dict = torch.load(args.first_step_dir)
        old_model.load_state_dict(old_state_dict, strict=False)

        # New Model
        model = ResNetTri(BasicBlock, [2, 2, 2, 2], args.num_labeled_classes,
                          args.num_unlabeled_classes1, args.num_unlabeled_classes2).to(device)
        state_dict = torch.load(args.model_dir)
        model.load_state_dict(state_dict, strict=False)



        # LOOK: OUR TEST FLOW
        # =============================== Final Test ===============================
        print("=" * 150)
        print("\t\t\t\tSecond: test function 1")
        print("=" * 150)

        acc_list = []
        args.head = 'head2'
        args.test_new = 'new1'
        print('Head2: test on unlabeled classes')
        _, ind1 = fair_test(old_model, p_unlabeled_val_loader, args, return_ind=True)

        args.head = 'head3'
        args.test_new = 'new2'
        print('Head3: test on unlabeled classes')
        _, ind2 = fair_test(model, unlabeled_val_loader, args, return_ind=True)

        args.head = 'head1'
        print('Evaluating on Head1')
        acc_all = 0.

        print('test on labeled classes (test split)')
        acc = fair_test(model, labeled_test_loader, args, cluster=False)
        acc_list.append(acc)
        acc_all = acc_all + acc * args.num_labeled_classes

        args.test_new = 'new1'
        print('test on unlabeled classes 1nd-NEW (test split)')
        acc = fair_test(model, p_unlabeled_test_loader, args, cluster=False, ind=ind1)
        acc_list.append(acc)
        acc_all = acc_all + acc * args.num_unlabeled_classes1

        args.test_new = 'new2'
        print('test on unlabeled classes 2nd-NEW (test split)')
        acc = fair_test(model, unlabeled_test_loader, args, cluster=False, ind=ind2)
        acc_list.append(acc)
        acc_all = acc_all + acc * args.num_unlabeled_classes2


        print('test on all classes m/ clustering (test split)')
        acc = acc_all / num_classes
        acc_list.append(acc)

        print('test on all classes w/o clustering (test split)')
        acc = fair_test(model, all_test_loader, args, cluster=False, ind=ind2)
        acc_list.append(acc)

        args.head = 'head2'
        print('Evaluating on Head2')

        print('test on unlabeled classes (train split)')
        acc = fair_test(model, p_unlabeled_val_loader, args)
        acc_list.append(acc)

        print('test on unlabeled classes (test split)')
        acc = fair_test(model, p_unlabeled_test_loader, args)
        acc_list.append(acc)

        args.head = 'head3'
        print('Evaluating on Head3')

        print('test on unlabeled classes (train split)')
        acc = fair_test(model, unlabeled_val_loader, args)
        acc_list.append(acc)

        print('test on unlabeled classes (test split)')
        acc = fair_test(model, unlabeled_test_loader, args)
        acc_list.append(acc)

        print(
            'Acc List: Head1 -> Old, New-1, New-2, All_wo_cluster, All_w_cluster, Head2->Train, Test, Head3->Train, Test')
        print(acc_list)
    else:
        print("PASS--------------------")

