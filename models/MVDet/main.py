import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
import tqdm
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from multiview_detector.datasets import *
from multiview_detector.loss.gaussian_mse import GaussianMSE
from multiview_detector.loss.brl_gaussian_mse import BRLGaussianMSE
from multiview_detector.loss.pseudo_label import PseudoLabelLoss
from multiview_detector.loss.heatmap_loss import HeatmapLoss
from multiview_detector.models.persp_trans_detector import PerspTransDetector
from multiview_detector.models.image_proj_variant import ImageProjVariant
from multiview_detector.models.res_proj_variant import ResProjVariant
from multiview_detector.models.no_joint_conv_variant import NoJointConvVariant
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.trainer import PerspectiveTrainer

def build_criterion(args):
    if args.loss == 'brl':
        base = BRLGaussianMSE(
            pos_thr=args.brl_pos_thr,
            confuse_pred_thr=args.brl_confuse_thr,
            beta=args.brl_beta,
            mirror=not args.brl_no_mirror,
        )
    else:
        base = GaussianMSE()
    pseudo = None
    if args.use_pseudo_labels:
        pseudo = PseudoLabelLoss(
            peak_weight=args.pseudo_loss_weight,
            neg_reweight=not args.pseudo_no_neg_reweight,
            peak_loss=not args.pseudo_no_peak_loss,
        )
    return HeatmapLoss(base, pseudo).cuda()

def main(args):
    # seed
    if args.seed is not None:
        np.random.seed(args.seed)
        torch.manual_seed(args.seed)
        # torch.backends.cudnn.deterministic = True
        # torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.benchmark = True
    else:
        torch.backends.cudnn.benchmark = True

    # dataset
    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    train_trans = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize, ])
    if 'wildtrack' in args.dataset:
        data_path = os.path.expanduser(args.data_path or '~/Data/Wildtrack')
        base = Wildtrack(data_path)
    elif 'multiviewx' in args.dataset:
        data_path = os.path.expanduser(args.data_path or '~/Data/MultiviewX')
        base = MultiviewX(data_path)
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')
    
    if args.use_pseudo_labels and not args.pseudo_cache:
        raise ValueError('--use_pseudo_labels requires --pseudo_cache')
    train_set = frameDataset(base, train=True, transform=train_trans, grid_reduce=4, drop_ratio=args.drop_ratio,
                             drop_mode=args.drop_mode,
                             pseudo_cache=args.pseudo_cache if args.use_pseudo_labels else None,
                             pseudo_conf_threshold=args.pseudo_conf_threshold,
                             pseudo_sigma_m=args.pseudo_sigma_m,
                             pseudo_suppress_radius_m=args.pseudo_suppress_radius_m)
    test_set = frameDataset(base, train=False, transform=train_trans, grid_reduce=4, drop_ratio=args.drop_ratio,
                           drop_mode=args.drop_mode)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    # model
    if args.variant == 'default':
        model = PerspTransDetector(train_set, args.arch)
    elif args.variant == 'img_proj':
        model = ImageProjVariant(train_set, args.arch)
    elif args.variant == 'res_proj':
        model = ResProjVariant(train_set, args.arch)
    elif args.variant == 'no_joint_conv':
        model = NoJointConvVariant(train_set, args.arch)
    else:
        raise Exception('no support for this variant')

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs)

    # loss
    criterion = build_criterion(args)

    # logging
    drop_prefix = 'frame_drop' if args.drop_mode == 'frame' else 'drop'
    drop_tag = f'{drop_prefix}_{args.drop_ratio}' if args.drop_ratio > 0 else 'full'
    if args.loss == 'brl':
        loss_tag = f'brl_p{args.brl_pos_thr}_c{args.brl_confuse_thr}_b{args.brl_beta}'
        if args.brl_no_mirror:
            loss_tag += '_nomirror'
    else:
        loss_tag = 'mse'
    if args.use_pseudo_labels:
        loss_tag += (f'_pseudo_w{args.pseudo_loss_weight:g}_c{args.pseudo_conf_threshold:g}'
                     f'_s{args.pseudo_sigma_m:g}_r{args.pseudo_suppress_radius_m:g}')
        if args.pseudo_no_neg_reweight:
            loss_tag += '_noneg'
        if args.pseudo_no_peak_loss:
            loss_tag += '_nopeak'
    logdir = f'logs/{args.dataset}_frame/{drop_tag}/{loss_tag}/{args.variant}/' + datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S') \
        if not args.resume else f'logs/{args.dataset}_frame/{drop_tag}/{loss_tag}/{args.variant}/{args.resume}'
    if args.resume is None:
        os.makedirs(logdir, exist_ok=True)
        copy_tree('./multiview_detector', logdir + '/scripts/multiview_detector')
        for script in os.listdir('.'):
            if script.split('.')[-1] == 'py':
                dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)
        sys.stdout = Logger(os.path.join(logdir, 'log.txt'), )
    print('Settings:')
    print(vars(args))

    # draw curve
    x_epoch = []
    train_loss_s = []
    train_prec_s = []
    test_loss_s = []
    test_prec_s = []
    test_moda_s = []

    trainer = PerspectiveTrainer(model, criterion, logdir, denormalize, args.cls_thres, args.alpha)

    # learn
    if args.resume is None:
        print('Testing...')
        trainer.test(test_loader, os.path.join(logdir, 'test.txt'), train_set.gt_fpath, True)

        for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
            print('Training...')
            train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, args.log_interval, scheduler)
            print('Testing...')
            test_loss, test_prec, moda = trainer.test(test_loader, os.path.join(logdir, 'test.txt'),
                                                      train_set.gt_fpath, True)

            x_epoch.append(epoch)
            train_loss_s.append(train_loss)
            train_prec_s.append(train_prec)
            test_loss_s.append(test_loss)
            test_prec_s.append(test_prec)
            test_moda_s.append(moda)
            draw_curve(os.path.join(logdir, 'learning_curve.jpg'), x_epoch, train_loss_s, train_prec_s,
                       test_loss_s, test_prec_s, test_moda_s)
            # save
            torch.save(model.state_dict(), os.path.join(logdir, 'MultiviewDetector.pth'))
    else:
        resume_fname = logdir + '/MultiviewDetector.pth'
        model.load_state_dict(torch.load(resume_fname))
        model.eval()
    print('Test loaded model...')
    trainer.test(test_loader, os.path.join(logdir, 'test.txt'), train_set.gt_fpath, True)


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--variant', type=str, default='default',
                        choices=['default', 'img_proj', 'res_proj', 'no_joint_conv'])
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18'])
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx'])
    parser.add_argument('--data_path', type=str, default=None, help='Dataset root')
    parser.add_argument('-j', '--num_workers', type=int, default=4)
    parser.add_argument('-b', '--batch_size', type=int, default=1, metavar='N',
                        help='input batch size for training (default: 1)')
    parser.add_argument('--epochs', type=int, default=10, metavar='N', help='number of epochs to train (default: 10)')
    parser.add_argument('--lr', type=float, default=0.1, metavar='LR', help='learning rate (default: 0.1)')
    parser.add_argument('--weight_decay', type=float, default=5e-4)
    parser.add_argument('--momentum', type=float, default=0.5, metavar='M', help='SGD momentum (default: 0.5)')
    parser.add_argument('--log_interval', type=int, default=10, metavar='N',
                        help='how many batches to wait before logging training status')
    parser.add_argument('--resume', type=str, default=None)
    parser.add_argument('--visualize', action='store_true')
    parser.add_argument('--seed', type=int, default=1, help='random seed (default: None)')
    
    # Dropped annotations
    parser.add_argument('--drop_ratio', type=int, default=0,
                        choices=[0, 20, 45, 60],
                        help='0 = full labels; 20/45/60 = drop_annotations/drop_XX')
    parser.add_argument('--drop_mode', type=str, default='instance', choices=['instance', 'frame'],
                        help='instance = drop_annotations/drop_XX (people dropped per frame); '
                             'frame = drop_annotations/frame_drop_XX (whole train frames dropped)')
    
    # BRL heatmap loss
    parser.add_argument('--loss', type=str, default='brl', choices=['brl', 'mse'],
                        help='brl = Background Recalibration; mse = original GaussianMSE')
    parser.add_argument('--brl_pos_thr', type=float, default=0.1,
                        help='soft-GT threshold for positive pixels (brl v1 only)')
    parser.add_argument('--brl_confuse_thr', type=float, default=0.3,
                        help='pred threshold on non-GT to mark confuse (possible missing GT)')
    parser.add_argument('--brl_beta', type=float, default=0.1,
                        help='weight / strength of confuse term')
    parser.add_argument('--brl_no_mirror', action='store_true',
                        help='down-weight bg MSE on confuse pixels instead of mirroring toward 1')

    # Pseudo labels from an offline 2D detector (BEV map only, train set only)
    parser.add_argument('--use_pseudo_labels', action='store_true',
                        help='add offline detector evidence to the BEV training loss')
    parser.add_argument('--pseudo_cache', type=str, default=None,
                        help='JSON cache generated by generate_pseudo_cache.py')
    parser.add_argument('--pseudo_loss_weight', type=float, default=0.01,
                        help='weight of the pseudo-peak term (lambda); does not scale neg reweighting')
    parser.add_argument('--pseudo_conf_threshold', type=float, default=0.5,
                        help='min detector score for a pseudo label')
    parser.add_argument('--pseudo_sigma_m', type=float, default=0.5,
                        help='std of the pseudo BEV Gaussian in metres')
    parser.add_argument('--pseudo_suppress_radius_m', type=float, default=1.0,
                        help='drop pseudo labels within this radius of a kept GT')
    parser.add_argument('--pseudo_no_neg_reweight', action='store_true',
                        help='ablation: do not scale the GT background term by (1 - pseudo_weight)')
    parser.add_argument('--pseudo_no_peak_loss', action='store_true',
                        help='ablation: no pseudo-peak term (peaks stay in the GT background term)')

    args = parser.parse_args()

    main(args)
