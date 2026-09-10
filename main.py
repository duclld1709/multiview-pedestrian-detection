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
from multiview_detector.loss.brl_loss import BRLLoss
from multiview_detector.models.persp_trans_detector import PerspTransDetector
from multiview_detector.models.image_proj_variant import ImageProjVariant
from multiview_detector.models.res_proj_variant import ResProjVariant
from multiview_detector.models.no_joint_conv_variant import NoJointConvVariant
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.trainer import PerspectiveTrainer


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
    current_dir = os.path.dirname(os.path.abspath(__file__))
    data_path = os.path.join(os.path.dirname(current_dir), 'Dataset')

    if 'wildtrack' in args.dataset:
        base = Wildtrack(os.path.join(data_path, 'Wildtrack'))

    elif 'multiviewx' in args.dataset:
        base = MultiviewX(os.path.join(data_path, 'MultiviewX'))
    else:
        raise Exception('must choose from [wildtrack, multiviewx]')
    train_set = frameDataset(base, train=True, transform=train_trans, grid_reduce=4)
    test_set = frameDataset(base, train=False, transform=train_trans, grid_reduce=4)

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

    # __BRL_STABILITY_PATCHED__
    if args.loss_type == 'brl':
        import math

        def _init_focal_bias(module, pi):
            if hasattr(module, 'bias') and module.bias is not None:
                with torch.no_grad():
                    module.bias.fill_(-math.log((1 - pi) / pi))
                return True
            return False

        _ok_map = hasattr(model, 'map_classifier') and _init_focal_bias(model.map_classifier[-1], args.brl_prior)
        _ok_img = hasattr(model, 'img_classifier') and _init_focal_bias(model.img_classifier[-1], args.brl_prior)
        print(f'BRL bias-init: map_classifier={_ok_map}, img_classifier={_ok_img}, prior={args.brl_prior}')
        if not (_ok_map and _ok_img):
            print('  [CANH BAO] khong tim thay bias - chay lai patch model truoc.')

        if args.brl_auto_lr and args.lr == 0.1:
            print(f'BRL auto-lr: giam lr tu {args.lr} xuong 0.01 (lr=0.1 tune cho GaussianMSE). '
                  'Truyen --lr <gia tri> hoac --brl_auto_lr 0 de tat.')
            args.lr = 0.01

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs)

    # loss
    if args.loss_type == 'brl':
        criterion = BRLLoss(alpha=args.brl_alpha, gamma=args.brl_gamma,
                            conf_thres=args.brl_thres, recalib_weight=args.brl_weight,
                            warmup_epochs=args.brl_warmup).cuda()
        print(f'Using BRLLoss (t={args.brl_thres}, weight={args.brl_weight}, '
              f'warmup={args.brl_warmup} epochs)')
    else:
        criterion = GaussianMSE().cuda()
        print('Using original GaussianMSE loss')

    # logging
    logdir = f'logs/{args.dataset}_frame/{args.variant}/' + datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S') \
        if not args.resume else f'logs/{args.dataset}_frame/{args.variant}/{args.resume}'
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
        resume_dir = f'logs/{args.dataset}_frame/{args.variant}/' + args.resume
        resume_fname = resume_dir + '/MultiviewDetector.pth'
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
    parser.add_argument('--loss_type', type=str, default='brl', choices=['mse', 'brl'],
                        help='mse = GaussianMSE goc; brl = Background Recalibration Loss')
    parser.add_argument('--brl_alpha', type=float, default=0.25)
    parser.add_argument('--brl_gamma', type=float, default=2.0)
    parser.add_argument('--brl_thres', type=float, default=0.5,
                        help='nguong nghi ngo t: pixel nen co p >= 1-t se duoc recalibrate')
    parser.add_argument('--brl_weight', type=float, default=0.1,
                        help='he so scale nhanh recalibrated (paper goc: 0.1-0.25 la toi uu)')
    parser.add_argument('--brl_warmup', type=int, default=1,
                        help='so epoch dau dung focal loss thuan truoc khi bat recalibration')
    parser.add_argument('--brl_prior', type=float, default=0.01,
                        help='xac suat tien nghiem de khoi tao bias lop cuoi khi dung BRL')
    parser.add_argument('--brl_auto_lr', type=int, default=1, choices=[0, 1],
                        help='1 = tu giam lr khi dung BRL neu ban khong tu truyen --lr')
    args = parser.parse_args()

    main(args)
