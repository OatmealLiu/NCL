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
from utils.fair_evals import fair_test, cluster_acc


def train(model, train_loader, unlabeled_eval_loader, labeled_eval_loader, all_eval_loader, args):
    print("="*100)
    print("\t\t\t\t\t1st-step Training")
    print("="*100)

    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = BCE()

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()

        exp_lr_scheduler.step()
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)
        for batch_idx, ((x, x_bar),  label, idx) in enumerate(tqdm(train_loader)):

            x, x_bar, label = x.to(device), x_bar.to(device), label.to(device)
            idx = idx.to(device)
            # print(args.num_labeled_classes)
            mask_lb = label < args.num_labeled_classes
            feat, feat_q, output1, output2 = model(x, 'feat_logit')
            feat_bar, feat_k, output1_bar, output2_bar = model(x_bar, 'feat_logit')

            prob1, prob1_bar, prob2, prob2_bar = F.softmax(output1, dim=1), F.softmax(output1_bar, dim=1), F.softmax(
                output2, dim=1), F.softmax(output2_bar, dim=1)

            # print(feat.shape)
            rank_feat = (feat[~mask_lb]).detach()
            # print(rank_feat.shape)
            if args.bce_type == 'cos':
                # default: cosine similarity with threshold
                feat_row, feat_col = PairEnum(F.normalize(rank_feat, dim=1))
                # print(feat_row.shape)
                # print(feat_col.shape)
                tmp_distance_ori = torch.bmm(feat_row.view(feat_row.size(0), 1, -1), feat_col.view(feat_row.size(0), -1, 1))
                tmp_distance_ori = tmp_distance_ori.squeeze()
                target_ulb = torch.zeros_like(tmp_distance_ori).float() - 1
                target_ulb[tmp_distance_ori > args.costhre] = 1
            elif args.bce_type == 'RK':
                # top-k rank statics
                rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
                rank_idx1, rank_idx2 = PairEnum(rank_idx)
                rank_idx1, rank_idx2 = rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]
                rank_idx1, _ = torch.sort(rank_idx1, dim=1)
                rank_idx2, _ = torch.sort(rank_idx2, dim=1)
                rank_diff = rank_idx1 - rank_idx2
                rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
                target_ulb = torch.ones_like(rank_diff).float().to(device)
                target_ulb[rank_diff > 0] = -1

            prob1_ulb, _ = PairEnum(prob2[~mask_lb])
            _, prob2_ulb = PairEnum(prob2_bar[~mask_lb])

            # basic loss
            loss_ce = criterion1(output1[mask_lb], label[mask_lb])
            loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)
            consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob2, prob2_bar)
            loss = 0.
            if loss_ce==loss_ce:
                # print("------>>>>{}".format(loss_ce))
                loss += loss_ce
            if loss_bce==loss_bce:
                loss += loss_bce
            if consistency_loss==loss_bce:
                loss += w * consistency_loss
            # loss = loss_ce + loss_bce + w * consistency_loss

            # NCL loss for unlabeled data
            if feat_q[~mask_lb].shape[0] == 0 or feat_k[~mask_lb].shape[0] == 0:
                loss_ncl_ulb = 0
            elif feat_q[~mask_lb].shape[0] == 1 or feat_k[~mask_lb].shape[0] == 1:
                loss_ncl_ulb = 0
            else:
                loss_ncl_ulb = ncl_ulb(feat_q[~mask_lb], feat_k[~mask_lb], label[~mask_lb], epoch, False, ncl_la.memory.clone().detach())

            # NCL loss for labeled data
            if feat_q[mask_lb].shape[0] == 0 or feat_k[mask_lb].shape[0] == 0:
                loss_ncl_la = 0
            else:
                loss_ncl_la = ncl_la(feat_q[mask_lb], feat_k[mask_lb], label[mask_lb], epoch, True)

            if epoch > 0:
                loss += loss_ncl_ulb * args.w_ncl_ulb + loss_ncl_la * args.w_ncl_la
            else:
                loss += loss_ncl_la * args.w_ncl_la

            # ===================backward=====================
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # wandb loss logging
        wandb.log({"loss/total_loss": loss_record.avg}, step=epoch)
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))

        # LOOK: Let's use our evaluation to test their model
        # validation for unlabeled data with Backbone(on-the-fly) + head-2(on-the-fly)
        print('Head2: test on unlabeled classes')
        args.head = 'head2'
        acc_head2_ul, ind = fair_test(model, unlabeled_eval_loader, args, return_ind=True)

        # validation for labeled data with Backbone(on-the-fly) + head-1(frozen)
        print('Head1: test on labeled classes')
        args.head = 'head1'
        acc_head1_lb = fair_test(model, labeled_eval_loader, args, cluster=False)

        # validation for unlabeled data with Backbone(on-the-fly) + head-1(frozen)
        print('Head1: test on unlabeled classes')
        acc_head1_ul = fair_test(model, unlabeled_eval_loader, args, cluster=False, ind=ind)

        # validation for all
        print('Head1: test on all classes w/o clustering')
        acc_head1_all_wo_cluster = fair_test(model, all_eval_loader, args, cluster=False, ind=ind)

        print('Head1: test on all classes w/ clustering')
        acc_head1_all_w_cluster = fair_test(model, all_eval_loader, args, cluster=True)

        # wandb metrics logging
        wandb.log({
            "val_acc/head2_ul": acc_head2_ul,
            "val_acc/head1_lb": acc_head1_lb,
            "val_acc/head1_ul": acc_head1_ul,
            "val_acc/head1_all_wo_clutering": acc_head1_all_wo_cluster,
            "val_acc/head1_all_w_clustering": acc_head1_all_w_cluster
        }, step=epoch)
        # LOOK: our method ends

def train_second(model, train_loader, unlabeled_eval_loader, labeled_eval_loader, all_eval_loader,
                 p_unlabeled_eval_loader, args):
    print("="*100)
    print("\t\t\t\t\t2nd-step Training")
    print("="*100)
    optimizer = SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    exp_lr_scheduler = lr_scheduler.StepLR(optimizer, step_size=args.step_size, gamma=args.gamma)
    criterion1 = nn.CrossEntropyLoss()
    criterion2 = BCE()

    for epoch in range(args.epochs):
        loss_record = AverageMeter()
        model.train()
        exp_lr_scheduler.step()
        w = args.rampup_coefficient * ramps.sigmoid_rampup(epoch, args.rampup_length)
        for batch_idx, ((x, x_bar),  label, idx) in enumerate(tqdm(train_loader)):
            x, x_bar, label = x.to(device), x_bar.to(device), label.to(device)
            idx = idx.to(device)
            mask_lb = label < args.num_labeled_classes
            # print(mask_lb)

            feat, feat_q, output1, output2 = model(x, 'feat_logit')
            feat_bar, feat_k, output1_bar, output2_bar = model(x_bar, 'feat_logit')

            prob1, prob1_bar, prob2, prob2_bar = F.softmax(output1, dim=1), F.softmax(output1_bar, dim=1), F.softmax(
                output2, dim=1), F.softmax(output2_bar, dim=1)

            rank_feat = (feat[~mask_lb]).detach()
            if args.bce_type == 'cos':
                # default: cosine similarity with threshold
                feat_row, feat_col = PairEnum(F.normalize(rank_feat, dim=1))
                tmp_distance_ori = torch.bmm(feat_row.view(feat_row.size(0), 1, -1), feat_col.view(feat_row.size(0), -1, 1))
                tmp_distance_ori = tmp_distance_ori.squeeze()
                target_ulb = torch.zeros_like(tmp_distance_ori).float() - 1
                target_ulb[tmp_distance_ori > args.costhre] = 1
            elif args.bce_type == 'RK':
                # top-k rank statics
                rank_idx = torch.argsort(rank_feat, dim=1, descending=True)
                rank_idx1, rank_idx2 = PairEnum(rank_idx)
                rank_idx1, rank_idx2 = rank_idx1[:, :args.topk], rank_idx2[:, :args.topk]
                rank_idx1, _ = torch.sort(rank_idx1, dim=1)
                rank_idx2, _ = torch.sort(rank_idx2, dim=1)
                rank_diff = rank_idx1 - rank_idx2
                rank_diff = torch.sum(torch.abs(rank_diff), dim=1)
                target_ulb = torch.ones_like(rank_diff).float().to(device)
                target_ulb[rank_diff > 0] = -1

            prob1_ulb, _ = PairEnum(prob2[~mask_lb])
            _, prob2_ulb = PairEnum(prob2_bar[~mask_lb])

            # basic loss
            loss_ce = criterion1(output1[mask_lb], label[mask_lb])
            loss_bce = criterion2(prob1_ulb, prob2_ulb, target_ulb)
            consistency_loss = F.mse_loss(prob1, prob1_bar) + F.mse_loss(prob2, prob2_bar)
            loss = 0.
            if loss_ce==loss_ce:
                # print("------>>>>{}".format(loss_ce))
                loss += loss_ce
            if loss_bce==loss_bce:
                loss += loss_bce
            if consistency_loss==loss_bce:
                loss += w * consistency_loss
            # loss = loss_ce + loss_bce + w * consistency_loss

            if feat_q[~mask_lb].shape[0] == 0 or feat_k[~mask_lb].shape[0] == 0:
                loss_ncl_ulb = 0
            else:
                loss_ncl_ulb = ncl_ulb(feat_q[~mask_lb], feat_k[~mask_lb], label[~mask_lb], epoch, False, ncl_la.memory.clone().detach())

            # NCL loss for labeled data
            if feat_q[mask_lb].shape[0] == 0 or feat_k[mask_lb].shape[0] == 0:
                loss_ncl_la = 0
            else:
                loss_ncl_la = ncl_la(feat_q[mask_lb], feat_k[mask_lb], label[mask_lb], epoch, True)

            if epoch > 0:
                loss += loss_ncl_ulb * args.w_ncl_ulb + loss_ncl_la * args.w_ncl_la
            else:
                loss += loss_ncl_la * args.w_ncl_la

            # ===================backward=====================
            loss_record.update(loss.item(), x.size(0))
            optimizer.zero_grad()
            loss.backward()
            optimizer.step()

        # wandb loss logging
        wandb.log({"loss/total_loss": loss_record.avg}, step=epoch)
        print('Train Epoch: {} Avg Loss: {:.4f}'.format(epoch, loss_record.avg))

        # LOOK: Let's use our evaluation to test their model
        # validation for unlabeled data with Backbone(on-the-fly) + head-2(on-the-fly)
        args.head = 'head2'
        print('Head2: test on PRE-unlabeled classes')
        acc_head2_ul, ind2 = fair_test(model, p_unlabeled_eval_loader, args, return_ind=True)

        args.head = 'head3'
        print('Head3: test on unlabeled classes')
        acc_head3_ul, ind3 = fair_test(model, unlabeled_eval_loader, args, return_ind=True)

        # validation for labeled data with Backbone(on-the-fly) + head-1(frozen)
        args.head = 'head1'
        print('Head1: test on labeled classes')
        acc_head1_lb = fair_test(model, labeled_eval_loader, args, cluster=False)

        # validation for unlabeled data with Backbone(on-the-fly) + head-1(frozen)
        print('Head1: test on PRE-unlabeled classes')
        acc_head1_ul1 = fair_test(model, p_unlabeled_eval_loader, args, cluster=False, ind=ind2)

        # validation for unlabeled data with Backbone(on-the-fly) + head-1(frozen)
        print('Head1: test on CRT-unlabeled classes')
        acc_head1_ul2 = fair_test(model, unlabeled_eval_loader, args, cluster=False, ind=ind3)

        # validation for all
        print('Head1: test on all classes w/o clustering')
        acc_head1_all_wo_cluster = fair_test(model, all_eval_loader, args, cluster=False, ind=ind3)

        print('Head1: test on all classes w/ clustering')
        acc_head1_all_w_cluster = fair_test(model, all_eval_loader, args, cluster=True)

        # wandb metrics logging
        wandb.log({
            "val_acc/head2_ul": acc_head2_ul,
            "val_acc/head3_ul": acc_head3_ul,
            "val_acc/head1_lb": acc_head1_lb,
            "val_acc/head1_ul_1": acc_head1_ul1,
            "val_acc/head1_ul_2": acc_head1_ul2,
            "val_acc/head1_all_wo_clutering": acc_head1_all_wo_cluster,
            "val_acc/head1_all_w_clustering": acc_head1_all_w_cluster
        }, step=epoch)
        # LOOK: our method ends

def test(model, test_loader, args):
    model.eval()
    preds=np.array([])
    targets=np.array([])
    for batch_idx, (x, label, _) in enumerate(tqdm(test_loader)):
        x, label = x.to(device), label.to(device)
        output1, output2 = model(x)
        if args.head == 'head1':
            output = output1
        else:
            output = output2
        _, pred = output.max(1)
        targets = np.append(targets, label.cpu().numpy())
        preds = np.append(preds, pred.cpu().numpy())
    acc, nmi, ari = cluster_acc(targets.astype(int), preds.astype(int)), nmi_score(targets, preds), ari_score(targets, preds)
    print('Test acc {:.4f}, nmi {:.4f}, ari {:.4f}'.format(acc, nmi, ari))
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
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'])
    parser.add_argument('--wandb_entity', type=str, default='unitn-mhug')
    parser.add_argument('--step', type=str, default='first', choices=['first', 'second'])
    parser.add_argument('--first_step_dir', type=str,
                        default='./results/two_incd_cifar100_DTC/DTC_cifar100_incd_resnet18_80.pth')

    args = parser.parse_args()
    args.cuda = torch.cuda.is_available()
    args.device = torch.device("cuda" if args.cuda else "cpu")
    device = torch.device("cuda" if args.cuda else "cpu")
    torch.backends.cudnn.benchmark = True
    seed_torch(args.seed)
    runner_name = os.path.basename(__file__).split(".")[0]
    model_dir = os.path.join(args.exp_root, runner_name)
    if not os.path.exists(model_dir):
        os.makedirs(model_dir)
    args.model_dir = model_dir+'/'+ args.step + '_'+'{}.pth'.format(args.model_name)

    # WandB setting
    # use wandb logging
    wandb_run_name = args.model_name + '_NCL_supervised_' + str(args.seed)
    wandb.init(project='incd_dev_miu',
               entity=args.wandb_entity,
               name=wandb_run_name,
               mode=args.wandb_mode)

    if args.mode == 'train' and args.step == 'first':
        num_classes = args.num_labeled_classes + args.num_unlabeled_classes1
        mix_train_loader = CIFAR100LoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='train', aug='twice', shuffle=True, labeled_list=range(args.num_labeled_classes), unlabeled_list=range(args.num_labeled_classes, num_classes))

        # unlabeled_list=range(args.num_labeled_classes, num_classes))
        unlabeled_val_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                              aug=None,
                                              shuffle=False, target_list=range(args.num_labeled_classes, num_classes))
        unlabeled_test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                               aug=None,
                                               shuffle=False, target_list=range(args.num_labeled_classes, num_classes))
        labeled_test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None,
                                             shuffle=False, target_list=range(args.num_labeled_classes))
        all_test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None,
                                         shuffle=False, target_list=range(num_classes))

        ncl_ulb = NCLMemory(512, args.m_size, args.m_t, args.num_unlabeled_classes1, args.knn, args.w_pos,
                            args.hard_iter, args.num_hard, args.hard_negative_start).to(device)
        ncl_la = NCLMemory(512, args.m_size, args.m_t, args.num_labeled_classes, args.knn, args.w_pos, args.hard_iter,
                           args.num_hard, args.hard_negative_start).to(device)

        model = ResNet(BasicBlock, [2, 2, 2, 2], args.num_labeled_classes,
                       args.num_unlabeled_classes1+args.num_unlabeled_classes2).to(device)

        def copy_param(model, pretrain_dir):
            pre_dict = torch.load(pretrain_dir)
            new = list(pre_dict.items())
            dict_len = len(pre_dict.items())
            model_kvpair = model.state_dict()
            count = 0
            for count in range(dict_len):
                layer_name, weights = new[count]
                if 'contrastive_head' not in layer_name and 'shortcut' not in layer_name:
                    if 'backbone' in layer_name:
                        model_kvpair[layer_name[9:]] = weights
                    # else:
                    #     model_kvpair[layer_name] = weights
                    print(layer_name[9:])
                else:
                    continue
            model.load_state_dict(model_kvpair)
            return model


        state_dict = torch.load(args.warmup_model_dir)
        model.load_state_dict(state_dict, strict=False)
        model.head2 = nn.Linear(512, args.num_unlabeled_classes1).to(device)

        for name, param in model.named_parameters():
            if 'head' not in name and 'layer4' not in name:
                param.requires_grad = False

        train(model, mix_train_loader, unlabeled_val_loader, labeled_test_loader, all_test_loader, args)
        torch.save(model.state_dict(), args.model_dir)
        print("model saved to {}.".format(args.model_dir))
        # LOOK: OUR TEST FLOW
        # =============================== Final Test ===============================
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
    elif args.mode == 'train' and args.step == 'second':
        num_classes = args.num_labeled_classes + args.num_unlabeled_classes1 + args.num_unlabeled_classes2
        mix_train_loader = CIFAR100LoaderMix(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                             aug='twice', shuffle=True, labeled_list=range(args.num_labeled_classes),
                                             unlabeled_list=range(args.num_labeled_classes + args.num_unlabeled_classes1,
                                                                  num_classes))
        unlabeled_val_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                              aug=None, shuffle=False,
                                              target_list=range(args.num_labeled_classes + args.num_unlabeled_classes1,
                                                                num_classes))
        unlabeled_test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                               aug=None, shuffle=False,
                                               target_list=range(args.num_labeled_classes + args.num_unlabeled_classes1,
                                                                 num_classes))
        labeled_test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None,
                                             shuffle=False, target_list=range(args.num_labeled_classes))
        all_test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test', aug=None,
                                         shuffle=False, target_list=range(num_classes))

        # Previous step Novel classes dataloader
        p_unlabeled_val_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='train',
                                                aug=None, shuffle=False,
                                                target_list=range(args.num_labeled_classes,
                                                                  args.num_labeled_classes + args.num_unlabeled_classes1))
        p_unlabeled_test_loader = CIFAR100Loader(root=args.dataset_root, batch_size=args.batch_size, split='test',
                                                 aug=None, shuffle=False,
                                                 target_list=range(args.num_labeled_classes,
                                                                   args.num_labeled_classes + args.num_unlabeled_classes1))

        ncl_ulb = NCLMemory(512, args.m_size, args.m_t, args.num_unlabeled_classes1, args.knn, args.w_pos,
                            args.hard_iter, args.num_hard, args.hard_negative_start).to(device)
        ncl_la = NCLMemory(512, args.m_size, args.m_t, args.num_labeled_classes, args.knn, args.w_pos, args.hard_iter,
                           args.num_hard, args.hard_negative_start).to(device)

        model = ResNetTri(BasicBlock, [2, 2, 2, 2], args.num_labeled_classes,
                          args.num_unlabeled_classes1, args.num_unlabeled_classes2).to(device)


        def copy_param(model, pretrain_dir):
            pre_dict = torch.load(pretrain_dir)
            new = list(pre_dict.items())
            dict_len = len(pre_dict.items())
            model_kvpair = model.state_dict()
            count = 0
            for count in range(dict_len):
                layer_name, weights = new[count]
                if 'contrastive_head' not in layer_name and 'shortcut' not in layer_name:
                    if 'backbone' in layer_name:
                        model_kvpair[layer_name[9:]] = weights
                    # else:
                    #     model_kvpair[layer_name] = weights
                    print(layer_name[9:])
                else:
                    continue
            model.load_state_dict(model_kvpair)
            return model


        state_dict = torch.load(args.first_step_dir)
        model.load_state_dict(state_dict, strict=False)

        for name, param in model.named_parameters():
            if 'head' not in name and 'layer4' not in name:
                param.requires_grad = False

        train_second(model, mix_train_loader, unlabeled_val_loader, labeled_test_loader, all_test_loader,
                     p_unlabeled_val_loader, args)
        torch.save(model.state_dict(), args.model_dir)
        print("model saved to {}.".format(args.model_dir))

        # LOOK: OUR TEST FLOW
        # =============================== Final Test ===============================
        acc_list = []

        args.head = 'head2'
        print('Head2: test on unlabeled classes')
        _, ind1 = fair_test(model, p_unlabeled_val_loader, args, return_ind=True)

        args.head = 'head3'
        print('Head3: test on unlabeled classes')
        _, ind2 = fair_test(model, unlabeled_val_loader, args, return_ind=True)

        args.head = 'head1'
        print('Evaluating on Head1')

        print('test on labeled classes (test split)')
        acc = fair_test(model, labeled_test_loader, args, cluster=False)
        acc_list.append(acc)

        print('test on unlabeled classes 1nd-NEW (test split)')
        acc = fair_test(model, p_unlabeled_test_loader, args, cluster=False, ind=ind1)
        acc_list.append(acc)

        print('test on unlabeled classes 2nd-NEW (test split)')
        acc = fair_test(model, unlabeled_test_loader, args, cluster=False, ind=ind2)
        acc_list.append(acc)

        print('test on all classes w/o clustering (test split)')
        acc = fair_test(model, all_test_loader, args, cluster=False, ind=ind2)
        acc_list.append(acc)

        print('test on all classes w/ clustering (test split)')
        acc = fair_test(model, all_test_loader, args, cluster=True)
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

