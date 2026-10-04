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


def main(args):
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
        base = Wildtrack(os.path.expanduser('../Data/Wildtrack'))
    elif 'multiviewx' in args.dataset:
        base = MultiviewX(os.path.expanduser('../Data/MultiviewX'))
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')
    if args.use_pseudo_labels and not args.pseudo_cache:
        raise ValueError('--use_pseudo_labels requires --pseudo_cache')
    if args.use_pseudo_labels and args.use_mse:
        raise ValueError('pseudo labels are wired into the focal heatmap loss; disable --use_mse')
    train_set = frameDataset(base, train=True, world_reduce=args.world_reduce,
                             img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                             img_kernel_size=args.img_kernel_size, semi_supervised=args.semi_supervised,
                             dropout=args.dropcam, augmentation=args.augmentation, drop_ratio=args.drop_ratio,
                             pseudo_cache=args.pseudo_cache if args.use_pseudo_labels else None,
                             pseudo_conf_threshold=args.pseudo_conf_threshold,
                             pseudo_sigma_m=args.pseudo_sigma_m,
                             pseudo_suppress_radius_m=args.pseudo_suppress_radius_m)
    test_set = frameDataset(base, train=False, world_reduce=args.world_reduce,
                            img_reduce=args.img_reduce, world_kernel_size=args.world_kernel_size,
                            img_kernel_size=args.img_kernel_size, drop_ratio=args.drop_ratio)

    def seed_worker(worker_id):
        worker_seed = torch.initial_seed() % 2 ** 32
        np.random.seed(worker_seed)
        random.seed(worker_seed)

    train_loader = DataLoader(train_set, batch_size=args.batch_size, shuffle=True, num_workers=args.num_workers,
                              pin_memory=True, worker_init_fn=seed_worker)
    test_loader = DataLoader(test_set, batch_size=args.batch_size, shuffle=False, num_workers=args.num_workers,
                             pin_memory=True, worker_init_fn=seed_worker)

    # logging
    if args.resume is None:
        drop_tag = args.drop_ratio if args.drop_ratio else 'full'
        if args.apply_brl:  
            loss_tag = f'brl_c{args.brl_confuse_thr}_p{args.brl_pos_thr}_beta{args.brl_beta}'
        else:
            loss_tag = 'no_brl'
        if args.use_pseudo_labels:
            loss_tag += (f'_pseudo_w{args.pseudo_loss_weight:g}_c{args.pseudo_conf_threshold:g}'
                         f'_s{args.pseudo_sigma_m:g}_r{args.pseudo_suppress_radius_m:g}')
        logdir = f'logs/{args.dataset}/{drop_tag}/{loss_tag}/{"debug_" if is_debug else ""}{"SS_" if args.semi_supervised else ""}' \
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
        logdir = f'logs/{args.dataset}/{args.resume}'
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

    trainer = PerspectiveTrainer(
        model, logdir, args.cls_thres, args.alpha, args.use_mse, args.id_ratio,
        apply_brl=args.apply_brl,
        brl_confuse_thr=args.brl_confuse_thr,
        brl_beta=args.brl_beta,
        brl_pos_thr=args.brl_pos_thr,
        pseudo_loss_weight=args.pseudo_loss_weight if args.use_pseudo_labels else 0.0,
    )

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
        model.load_state_dict(torch.load(f'logs/{args.dataset}/{args.resume}/MultiviewDetector.pth'))
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
    parser.add_argument('--seed', type=int, default=2021, help='random seed')
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
    # Drop settings
    parser.add_argument('--drop_ratio', type=str, default=None,
                        choices=['drop_20', 'drop_45', 'drop_60'])
    # BRL settings
    parser.add_argument('--apply_brl', type=str2bool, default=False,
                        help='apply Background Recalibration Loss')
    parser.add_argument('--brl_pos_thr', type=float, default=0.1,
                        help='pixel with soft GT >= this is positive')
    parser.add_argument('--brl_confuse_thr', type=float, default=0.3,
                        help='pixel with prediction >= this is confuse (possible missing label)')
    parser.add_argument('--brl_beta', type=float, default=0.1,
                        help='weight for confuse term in BRLFocalLoss')
    # Pseudo-label settings (offline detector cache, BEV focal pseudo loss)
    parser.add_argument('--use_pseudo_labels', action='store_true',
                        help='supervise the BEV heatmap with projected detector pseudo labels')
    parser.add_argument('--pseudo_cache', type=str, default=None,
                        help='JSON cache generated by generate_pseudo_cache.py')
    parser.add_argument('--pseudo_loss_weight', type=float, default=0.1,
                        help='weight of the pseudo peak focal term relative to the BEV focal loss')
    parser.add_argument('--pseudo_conf_threshold', type=float, default=0.2)
    parser.add_argument('--pseudo_sigma_m', type=float, default=0.25,
                        help='pseudo blob sigma in meters (MVDeTr GT sigma is 0.25 m by default)')
    parser.add_argument('--pseudo_suppress_radius_m', type=float, default=1.0,
                        help='drop pseudo evidence within this radius of an annotated point')

    args = parser.parse_args()

    main(args)
