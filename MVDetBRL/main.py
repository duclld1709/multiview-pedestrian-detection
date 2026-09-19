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
from multiview_detector.loss.brl_noisy_gaussian_mse import BRLNoisyGaussianMSE
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
        return BRLGaussianMSE(
            pos_thr=args.brl_pos_thr,
            confuse_pred_thr=args.brl_confuse_thr,
            beta=args.brl_beta,
            mirror=not args.brl_no_mirror,
        ).cuda()
    if args.loss == 'brl_noise':
        return BRLNoisyGaussianMSE(
            pos_thr=args.brl_pos_thr,
            confuse_pred_thr=args.brl_confuse_thr,
            beta=args.brl_beta,
            mirror=not args.brl_no_mirror,
            noise_mode=args.noise_mode,
            noise_scope=args.noise_scope,
        ).cuda()
    return GaussianMSE().cuda()


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
        data_path = os.path.expanduser('/kaggle/working/Data_temp/Wildtrack')
        base = Wildtrack(data_path)
    elif 'multiviewx' in args.dataset:
        data_path = os.path.expanduser('/kaggle/working/Data_temp/MultiviewX')
        base = MultiviewX(data_path)
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')

    train_set = frameDataset(base, train=True, transform=train_trans, grid_reduce=4, drop_ratio=args.drop_ratio)
    test_set = frameDataset(base, train=False, transform=train_trans, grid_reduce=4, drop_ratio=args.drop_ratio)

    train_loader = torch.utils.data.DataLoader(train_set, batch_size=args.batch_size, shuffle=True,
                                               num_workers=args.num_workers, pin_memory=True)
    test_loader = torch.utils.data.DataLoader(test_set, batch_size=args.batch_size, shuffle=False,
                                              num_workers=args.num_workers, pin_memory=True)

    # model
    noise_kw = dict(use_noise=(args.loss == 'brl_noise'), noise_init_sigma=args.noise_init_sigma,
                    noise_sigma_min=args.noise_sigma_min)
    if args.variant == 'default':
        model = PerspTransDetector(train_set, args.arch, **noise_kw)
    elif args.variant == 'img_proj':
        model = ImageProjVariant(train_set, args.arch, **noise_kw)
    elif args.variant == 'res_proj':
        model = ResProjVariant(train_set, args.arch, **noise_kw)
    elif args.variant == 'no_joint_conv':
        model = NoJointConvVariant(train_set, args.arch, **noise_kw)
    else:
        raise Exception('no support for this variant')

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs)

    # loss
    criterion = build_criterion(args)

    # logging
    drop_tag = f'drop_{args.drop_ratio}' if args.drop_ratio > 0 else 'full'
    if args.loss == 'brl':
        loss_tag = f'brl_b{args.brl_beta}_c{args.brl_confuse_thr}'
        if args.brl_no_mirror:
            loss_tag += '_nomirror'
    elif args.loss == 'brl_noise':
        loss_tag = f'brlnoise_{args.noise_mode}_{args.noise_scope}_s{args.noise_init_sigma}_m{args.noise_sigma_min}'
    else:
        loss_tag = 'mse'
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
        resume_dir = f'logs/{args.dataset}_frame/{drop_tag}/{loss_tag}/{args.variant}/' + args.resume
        resume_fname = resume_dir + '/MultiviewDetector.pth'
        model.load_state_dict(torch.load(resume_fname))
        model.eval()
    print('Test loaded model...')
    trainer.test(test_loader, os.path.join(logdir, 'test.txt'), train_set.gt_fpath, True)


if __name__ == '__main__':
    # settings
    parser = argparse.ArgumentParser(description='Multiview detector + BRL heatmap loss')
    parser.add_argument('--reID', action='store_true')
    parser.add_argument('--cls_thres', type=float, default=0.4)
    parser.add_argument('--alpha', type=float, default=1.0, help='ratio for per view loss')
    parser.add_argument('--variant', type=str, default='default',
                        choices=['default', 'img_proj', 'res_proj', 'no_joint_conv'])
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18'])
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack', choices=['wildtrack', 'multiviewx'])
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
    # BRL heatmap loss
    parser.add_argument('--loss', type=str, default='brl', choices=['brl', 'brl_noise', 'mse'],
                        help='brl = Background Recalibration heatmap loss; '
                             'brl_noise = BRL + learned per-pixel noise (sigma head); mse = original GaussianMSE')
    parser.add_argument('--brl_pos_thr', type=float, default=0.1,
                        help='soft-GT threshold for positive pixels')
    parser.add_argument('--brl_confuse_thr', type=float, default=0.3,
                        help='pred threshold on background to mark confuse (possible missing GT)')
    parser.add_argument('--brl_beta', type=float, default=0.1,
                        help='weight / strength of confuse term')
    parser.add_argument('--brl_no_mirror', action='store_true',
                        help='if set, down-weight bg MSE on confuse instead of mirroring toward 1')
    # Learned-noise loss options, used only with --loss brl_noise.
    parser.add_argument('--noise_mode', type=str, default='nll', choices=['nll', 'prob'],
                        help='nll = Gaussian NLL (recommended); prob = MSE(x + sigma*eps, y) ablation')
    parser.add_argument('--noise_scope', type=str, default='confuse', choices=['confuse', 'all'],
                        help='confuse = noise only on confuse pixels; all = noise on every pixel')
    parser.add_argument('--noise_init_sigma', type=float, default=0.7)
    parser.add_argument('--noise_sigma_min', type=float, default=0.1)
    args = parser.parse_args()

    main(args)
