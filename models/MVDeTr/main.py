import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
import tqdm
import random
import numpy as np
import torch
from torch.cuda.amp import GradScaler
from torch import optim
from torch.utils.data import DataLoader
from multiview_detector.datasets import *
from multiview_detector.models.mvdetr import MVDeTr
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.str2bool import str2bool
from multiview_detector.trainer import PerspectiveTrainer
from multiview_detector.loss import FocalLoss, BRLFocalLoss, PseudoLabelLoss, HeatmapLoss


def build_focal_loss(args):
    if args.loss == 'brl':
        base = BRLFocalLoss(
            pos_thr=args.brl_pos_thr,
            confuse_pred_thr=args.brl_confuse_thr,
            beta=args.brl_beta,
            mirror=not args.brl_no_mirror,
        )
    else:
        base = FocalLoss()
    pseudo = None
    if args.use_pseudo_labels:
        pseudo = PseudoLabelLoss(
            peak_weight=args.pseudo_loss_weight,
            neg_reweight=not args.pseudo_no_neg_reweight,
            peak_loss=not args.pseudo_no_peak_loss,
        )
    return HeatmapLoss(base, pseudo)


def main(args):
    if args.use_mse and args.loss == 'brl':
        raise ValueError('--use_mse 1 replaces the focal heatmap loss, so BRL would be ignored; use --loss focal')
    if args.use_mse and args.use_pseudo_labels:
        raise ValueError('pseudo labels are wired into the focal heatmap loss; disable --use_mse')
    if args.use_pseudo_labels and not args.pseudo_cache:
        raise ValueError('--use_pseudo_labels requires --pseudo_cache')

    # check if in debug mode
    gettrace = getattr(sys, 'gettrace', None)
    if gettrace():
        print('Hmm, Big Debugger is watching me')
        is_debug = True
    else:
        print('No sys.gettrace')
        is_debug = False

    # seed
    if args.seed is not None:
        random.seed(args.seed)
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        torch.cuda.manual_seed(args.seed)
        torch.cuda.manual_seed_all(args.seed)

    # deterministic
    if args.deterministic:
        torch.backends.cudnn.deterministic = True
        torch.backends.cudnn.benchmark = False
        torch.autograd.set_detect_anomaly(True)
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    if 'wildtrack' in args.dataset:
        base = Wildtrack(os.path.expanduser(args.data_path or '~/Data/Wildtrack'))
    elif 'multiviewx' in args.dataset:
        base = MultiviewX(os.path.expanduser(args.data_path or '~/Data/MultiviewX'))
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')
    train_set = frameDataset(base, train=True, world_reduce=args.world_reduce,
                             img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                             img_kernel_size=args.img_kernel_size, semi_supervised=args.semi_supervised,
                             dropout=args.dropcam, augmentation=args.augmentation, drop_ratio=args.drop_ratio,
                             drop_mode=args.drop_mode,
                             pseudo_cache=args.pseudo_cache if args.use_pseudo_labels else None,
                             pseudo_conf_threshold=args.pseudo_conf_threshold,
                             pseudo_sigma_m=args.pseudo_sigma_m,
                             pseudo_suppress_radius_m=args.pseudo_suppress_radius_m)
    test_set = frameDataset(base, train=False, world_reduce=args.world_reduce,
                            img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                            img_kernel_size=args.img_kernel_size, drop_ratio=args.drop_ratio,
                            drop_mode=args.drop_mode)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, worker_init_fn=seed_worker)

    # logging
    drop_prefix = 'frame_drop' if args.drop_mode == 'frame' else 'drop'
    drop_tag = f'{drop_prefix}_{args.drop_ratio}' if args.drop_ratio > 0 else 'full'
    if args.use_mse:
        loss_tag = 'mse'
    elif args.loss == 'brl':
        loss_tag = f'brl_p{args.brl_pos_thr}_c{args.brl_confuse_thr}_b{args.brl_beta}'
        if args.brl_no_mirror:
            loss_tag += '_nomirror'
    else:
        loss_tag = 'focal'
    if args.use_pseudo_labels:
        loss_tag += (f'_pseudo_w{args.pseudo_loss_weight:g}_c{args.pseudo_conf_threshold:g}'
                     f'_s{args.pseudo_sigma_m:g}_r{args.pseudo_suppress_radius_m:g}')
        if args.pseudo_no_neg_reweight:
            loss_tag += '_noneg'
        if args.pseudo_no_peak_loss:
            loss_tag += '_nopeak'
    log_root = f'logs/{args.dataset}/{drop_tag}/{loss_tag}'
    if args.resume is None:
        logdir = f'{log_root}/{"debug_" if is_debug else ""}{"SS_" if args.semi_supervised else ""}' \
                 f'{"aug_" if args.augmentation else ""}{args.world_feat}_lr{args.lr}_baseR{args.base_lr_ratio}_' \
                 f'neck{args.bottleneck_dim}_out{args.outfeat_dim}_' \
                 f'alpha{args.alpha}_id{args.id_ratio}_drop{args.dropout}_dropcam{args.dropcam}_' \
                 f'worldRK{args.world_reduce}_{args.world_kernel_size}_imgRK{args.img_reduce}_{args.img_kernel_size}_' \
                 f'{datetime.datetime.today():%Y-%m-%d_%H-%M-%S}'
        os.makedirs(logdir, exist_ok=True)
        copy_tree('./multiview_detector', logdir + '/scripts/multiview_detector')
        for script in os.listdir('.'):
            if script.split('.')[-1] == 'py':
                dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)
        sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    else:
        logdir = f'{log_root}/{args.resume}'
    print(logdir)
    print('Settings:')
    print(vars(args))

    # model
    model = MVDeTr(train_set, args.arch, world_feat_arch=args.world_feat,
                   bottleneck_dim=args.bottleneck_dim, outfeat_dim=args.outfeat_dim, droupout=args.dropout).cuda()

    param_dicts = [{"params": [p for n, p in model.named_parameters() if 'base' not in n and p.requires_grad], },
                   {"params": [p for n, p in model.named_parameters() if 'base' in n and p.requires_grad],
                    "lr": args.lr * args.base_lr_ratio, }, ]
    # optimizer = optim.SGD(param_dicts, lr=args.lr, momentum=0.9, weight_decay=args.weight_decay)
    optimizer = optim.Adam(param_dicts, lr=args.lr, weight_decay=args.weight_decay)
    scaler = GradScaler()

    # def warmup_lr_scheduler(epoch, warmup_epochs=2):
    #     if epoch < warmup_epochs:
    #         return epoch / warmup_epochs
    #     else:
    #         return (np.cos((epoch - warmup_epochs) / (args.epochs - warmup_epochs) * np.pi) + 1) / 2

    # scheduler = torch.optim.lr_scheduler.CosineAnnealingWarmRestarts(optimizer, args.epochs)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs)
    # scheduler = torch.optim.lr_scheduler.MultiStepLR(optimizer, [10, 15], 0.1)
    # scheduler = torch.optim.lr_scheduler.LambdaLR(optimizer, warmup_lr_scheduler)

    trainer = PerspectiveTrainer(model, logdir, args.cls_thres, args.alpha, args.use_mse, args.id_ratio,
                                 focal_loss=build_focal_loss(args))

    # draw curve
    x_epoch = []
    train_loss_s = []
    test_loss_s = []
    test_moda_s = []

    # learn
    res_fpath = os.path.join(logdir, 'test.txt')
    if args.resume is None:
        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            print('Training...')
            train_loss = trainer.train(epoch, train_loader, optimizer, scaler, scheduler)
            print('Testing...')
            test_loss, moda = trainer.test(epoch, test_loader, res_fpath, visualize=True)

            # draw & save
            x_epoch.append(epoch)
            train_loss_s.append(train_loss)
            test_loss_s.append(test_loss)
            test_moda_s.append(moda)
            draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, test_loss_s, test_moda_s)
            torch.save(model.state_dict(), os.path.join(logdir, 'MultiviewDetector.pth'))
    else:
        model.load_state_dict(torch.load(os.path.join(logdir, 'MultiviewDetector.pth')))
        model.eval()
    print('Test loaded model...')
    trainer.test(None, test_loader, res_fpath, visualize=True)


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--semi_supervised', type=float, default=0)
    parser.add_argument('--id_ratio', type=float, default=0)
    parser.add_argument('--cls_thres', type=float, default=0.6)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--use_mse', type=str2bool, default=False)
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18', 'mobilenet'])
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx'])
    parser.add_argument('--data_path', type=str, default=None, help='Dataset root (default ~/Data/<dataset>)')
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1, help='input batch size for training')
    parser.add_argument('--dropout', type=float, default=0.0)
    parser.add_argument('--dropcam', type=float, default=0.0)
    parser.add_argument('--epochs', type=int, default=10, help='number of epochs to train')
    parser.add_argument('--lr', type=float, default=5e-4, help='learning rate')
    parser.add_argument('--base_lr_ratio', type=float, default=0.1)
    parser.add_argument('--weight_decay', type=float, default=1e-4)
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed')
    parser.add_argument('--deterministic', type=str2bool, default=False)
    parser.add_argument('--augmentation', type=str2bool, default=True)

    parser.add_argument('--world_feat', type=str, default='deform_trans',
                        choices=['conv', 'trans', 'deform_conv', 'deform_trans', 'aio'])
    parser.add_argument('--bottleneck_dim', type=int, default=128)
    parser.add_argument('--outfeat_dim', type=int, default=0)
    parser.add_argument('--world_reduce', type=int, default=4)
    parser.add_argument('--world_kernel_size', type=int, default=10)
    parser.add_argument('--img_reduce', type=int, default=12)
    parser.add_argument('--img_kernel_size', type=int, default=10)

    # Dropped annotations
    parser.add_argument('--drop_ratio', type=int, default=0, choices=[0, 20, 45, 60],
                        help='0 = full labels; 20/45/60 = drop_annotations/drop_XX')
    parser.add_argument('--drop_mode', type=str, default='instance', choices=['instance', 'frame'],
                        help='instance = drop_annotations/drop_XX (people dropped per frame); '
                             'frame = drop_annotations/frame_drop_XX (whole train frames dropped)')

    # BRL heatmap loss
    parser.add_argument('--loss', type=str, default='brl', choices=['brl', 'focal'],
                        help='brl = Background Recalibration on the focal loss; focal = original CornerNet focal loss')
    parser.add_argument('--brl_pos_thr', type=float, default=0.1,
                        help='GT heatmap below this is background (eligible for confuse)')
    parser.add_argument('--brl_confuse_thr', type=float, default=0.4,
                        help='pred threshold on background to mark confuse (possible missing GT)')
    parser.add_argument('--brl_beta', type=float, default=0.05,
                        help='weight / strength of confuse term')
    parser.add_argument('--brl_no_mirror', action='store_true',
                        help='down-weight the focal negative on confuse pixels instead of mirroring toward 1')

    # Pseudo labels from an offline 2D detector (BEV map only, train set only)
    parser.add_argument('--use_pseudo_labels', action='store_true',
                        help='add offline detector evidence to the BEV training loss')
    parser.add_argument('--pseudo_cache', type=str, default=None,
                        help='JSON detection cache (same format for every model)')
    parser.add_argument('--pseudo_loss_weight', type=float, default=0.01,
                        help='weight of the pseudo-peak term (lambda); does not scale neg reweighting')
    parser.add_argument('--pseudo_conf_threshold', type=float, default=0.5,
                        help='min detector score for a pseudo label')
    parser.add_argument('--pseudo_sigma_m', type=float, default=0.5,
                        help='std of the pseudo BEV Gaussian in metres (2x the 0.25 m GT kernel)')
    parser.add_argument('--pseudo_suppress_radius_m', type=float, default=1.0,
                        help='drop pseudo labels within this radius of a kept GT')
    parser.add_argument('--pseudo_no_neg_reweight', action='store_true',
                        help='ablation: do not scale the GT negative term by (1 - pseudo_weight)')
    parser.add_argument('--pseudo_no_peak_loss', action='store_true',
                        help='ablation: no pseudo-peak term (peaks stay in the GT negative term)')

    args = parser.parse_args()

    main(args)
