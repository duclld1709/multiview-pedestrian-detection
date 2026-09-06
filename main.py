import os

os.environ['OMP_NUM_THREADS'] = '1'
import argparse
import sys
import shutil
from distutils.dir_util import copy_tree
import datetime
from pathlib import Path
import tqdm
import numpy as np
import torch
import torch.optim as optim
import torchvision.transforms as T
from dotenv import load_dotenv
from multiview_detector.datasets import *
from multiview_detector.loss.gaussian_mse import GaussianMSE
from multiview_detector.models.persp_trans_detector import PerspTransDetector
from multiview_detector.models.image_proj_variant import ImageProjVariant
from multiview_detector.models.res_proj_variant import ResProjVariant
from multiview_detector.models.no_joint_conv_variant import NoJointConvVariant
from multiview_detector.utils.logger import Logger
from multiview_detector.utils.draw_curve import draw_curve
from multiview_detector.utils.image_utils import img_color_denormalize
from multiview_detector.trainer import PerspectiveTrainer


WANDB_ENTITY = 'GFA26AI02'
WANDB_PROJECT = 'baseline-expriments'


def init_wandb(args, model, train_set, test_set, logdir):
    """Initialize one W&B run for either Wildtrack or MultiviewX."""
    try:
        import wandb
    except ImportError as exc:
        raise RuntimeError(
            'Cannot import Weights & Biases. Reinstall the W&B dependencies from requirements.txt.'
        ) from exc

    load_dotenv(Path(__file__).resolve().with_name('.env'))

    # W&B expects WANDB_API_KEY. Keep compatibility with the WANDB_API name
    # already used by this project without ever adding the secret to config.
    if not os.getenv('WANDB_API_KEY') and os.getenv('WANDB_API'):
        os.environ['WANDB_API_KEY'] = os.environ['WANDB_API']

    timestamp = datetime.datetime.now().strftime('%Y%m%d-%H%M%S')
    job_type = 'inference' if args.resume else 'train'
    run_name = args.wandb_run_name or f'{args.dataset}-{args.variant}-{args.arch}-{job_type}-{timestamp}'
    config = dict(vars(args))
    config.update({
        'model_name': type(model).__name__,
        'model_variant': args.variant,
        'architecture': args.arch,
        'dataset_name': args.dataset,
        'optimizer': 'SGD',
        'scheduler': 'OneCycleLR',
        'learning_rate': args.lr,
        'momentum': args.momentum,
        'weight_decay': args.weight_decay,
        'batch_size': args.batch_size,
        'epochs': args.epochs,
        'alpha': args.alpha,
        'classification_threshold': args.cls_thres,
        'num_workers': args.num_workers,
        'seed': args.seed,
        'grid_reduce': train_set.grid_reduce,
        'image_reduce': train_set.img_reduce,
        'train_ratio': 0.9,
        'input_size': [720, 1280],
        'num_cameras': train_set.num_cam,
        'train_samples': len(train_set),
        'test_samples': len(test_set),
        # No partial-annotation pipeline exists yet. Set this to the real drop
        # ratio when that pipeline is added, rather than silently claiming 0%.
        'annotation_drop_rate': getattr(args, 'annotation_drop_rate', None),
        'resume_checkpoint': args.resume,
    })
    run = wandb.init(
        entity=WANDB_ENTITY,
        project=WANDB_PROJECT,
        name=run_name,
        job_type=job_type,
        config=config,
        dir=logdir,
        mode=args.wandb_mode,
        tags=[args.dataset, args.variant, args.arch],
    )
    run.define_metric('epoch')
    for namespace in ('train/*', 'validation/*', 'final_test/*', 'gpu/*'):
        run.define_metric(namespace, step_metric='epoch')
    return run


def gpu_metrics():
    """Return explicit per-GPU memory/utilization metrics for W&B and stdout."""
    if not torch.cuda.is_available():
        return {'gpu/available': 0}

    metrics = {'gpu/available': 1, 'gpu/device_count': torch.cuda.device_count()}
    for device_idx in range(torch.cuda.device_count()):
        prefix = f'gpu/{device_idx}'
        props = torch.cuda.get_device_properties(device_idx)
        allocated = torch.cuda.memory_allocated(device_idx)
        reserved = torch.cuda.memory_reserved(device_idx)
        max_allocated = torch.cuda.max_memory_allocated(device_idx)
        metrics[f'{prefix}/memory_allocated_mb'] = allocated / (1024 ** 2)
        metrics[f'{prefix}/memory_reserved_mb'] = reserved / (1024 ** 2)
        metrics[f'{prefix}/max_memory_allocated_mb'] = max_allocated / (1024 ** 2)
        metrics[f'{prefix}/memory_allocated_percent'] = allocated / props.total_memory * 100
        try:
            metrics[f'{prefix}/utilization_percent'] = torch.cuda.utilization(device_idx)
        except Exception:
            # W&B's system monitor still records utilization when NVML is available.
            pass
    return metrics


def reset_peak_gpu_memory():
    if torch.cuda.is_available():
        for device_idx in range(torch.cuda.device_count()):
            torch.cuda.reset_peak_memory_stats(device_idx)


def log_phase(run, phase, epoch, metrics, optimizer=None):
    payload = {'epoch': epoch}
    payload.update({f'{phase}/{key}': value for key, value in metrics.items()})
    payload.update(gpu_metrics())
    if optimizer is not None:
        payload['train/learning_rate'] = optimizer.param_groups[0]['lr']
    run.log(payload)

    gpu_text = ', '.join(
        f'GPU {idx}: {payload.get(f"gpu/{idx}/utilization_percent", "n/a")}% util, '
        f'{payload[f"gpu/{idx}/memory_allocated_mb"]:.0f} MB allocated'
        for idx in range(torch.cuda.device_count())
    ) if torch.cuda.is_available() else 'GPU: unavailable'
    metric_text = ', '.join(f'{key}: {value:.4f}' for key, value in metrics.items())
    print(f'[{phase}] Epoch: {epoch}, {metric_text}, {gpu_text}')


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
        data_path = os.path.expanduser('~/Data/Wildtrack')
        base = Wildtrack(data_path)
    elif 'multiviewx' in args.dataset:
        data_path = os.path.expanduser('~/Data/MultiviewX')
        base = MultiviewX(data_path)
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

    optimizer = optim.SGD(model.parameters(), lr=args.lr, momentum=args.momentum, weight_decay=args.weight_decay)
    scheduler = torch.optim.lr_scheduler.OneCycleLR(optimizer, max_lr=args.lr, steps_per_epoch=len(train_loader),
                                                    epochs=args.epochs)

    # loss
    criterion = GaussianMSE().cuda()

    # local and W&B logging
    logdir = f'logs/{args.dataset}_frame/{args.variant}/' + datetime.datetime.today().strftime('%Y-%m-%d_%H-%M-%S') \
        if not args.resume else f'logs/{args.dataset}_frame/{args.variant}/{args.resume}'
    if args.resume is None:
        os.makedirs(logdir, exist_ok=True)
        copy_tree('./multiview_detector', logdir + '/scripts/multiview_detector')
        for script in os.listdir('.'):
            if script.split('.')[-1] == 'py':
                dst_file = os.path.join(logdir, 'scripts', os.path.basename(script))
                shutil.copyfile(script, dst_file)

    wandb_run = init_wandb(args, model, train_set, test_set, logdir)

    if args.resume is None:
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
    try:
        if args.resume is None:
            print('Testing before training...')
            reset_peak_gpu_memory()
            trainer.test(test_loader, os.path.join(logdir, 'test.txt'), train_set.gt_fpath, True)
            log_phase(wandb_run, 'validation', 0, trainer.last_test_metrics)

            for epoch in tqdm.tqdm(range(1, args.epochs + 1)):
                print('Training...')
                reset_peak_gpu_memory()
                train_loss, train_prec = trainer.train(epoch, train_loader, optimizer, args.log_interval, scheduler)
                log_phase(wandb_run, 'train', epoch, trainer.last_train_metrics, optimizer)

                print('Testing...')
                reset_peak_gpu_memory()
                test_loss, test_prec, moda = trainer.test(test_loader, os.path.join(logdir, 'test.txt'),
                                                          train_set.gt_fpath, True)
                log_phase(wandb_run, 'validation', epoch, trainer.last_test_metrics)

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
        reset_peak_gpu_memory()
        trainer.test(test_loader, os.path.join(logdir, 'test.txt'), train_set.gt_fpath, True)
        final_epoch = args.epochs if args.resume is None else 0
        log_phase(wandb_run, 'final_test', final_epoch, trainer.last_test_metrics)
    finally:
        wandb_run.finish()


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
    parser.add_argument('--wandb_run_name', type=str, default=None, help='optional custom W&B run name')
    parser.add_argument('--wandb_mode', type=str, default='online', choices=['online', 'offline', 'disabled'],
                        help='W&B sync mode (default: online)')
    args = parser.parse_args()

    main(args)
