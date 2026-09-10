#!/usr/bin/env python3
# -*- coding: utf-8 -*-
"""
DEMO SUY LUẬN (INFERENCE) CHO MVDet
"Multiview Detection with Feature Perspective Transformation" (ECCV 2020)
Repo gốc: https://github.com/hou-yz/MVDet

Script này KHÔNG train, chỉ nạp checkpoint đã có sẵn (MultiviewDetector.pth) và
chạy thử mô hình trên 1 (hoặc vài) frame của tập test để:
  1) In ra danh sách vị trí người đi bộ được phát hiện (toạ độ mặt đất, đơn vị mét)
  2) Vẽ bản đồ occupancy (heatmap) dự đoán vs. ground-truth (nhìn từ trên xuống)
  3) Vẽ heatmap "head/foot" chồng lên ảnh gốc của từng camera
  4) Tính nhanh precision/recall theo ngưỡng (không cần MATLAB)

QUAN TRỌNG:
  - Model gốc trong `multiview_detector/models/persp_trans_detector.py` bị hard-code
    chạy trên 2 GPU (`cuda:0` và `cuda:1`). Script này định nghĩa lại kiến trúc y hệt
    nhưng chạy trên MỘT thiết bị (CPU hoặc 1 GPU) để tiện demo trên máy cá nhân.
    Vì tên các layer giữ nguyên, checkpoint gốc (MultiviewDetector.pth) vẫn load
    thẳng vào được, không cần train lại.

CÁCH DÙNG (chạy từ thư mục gốc của repo MVDet, nơi có sẵn thư mục multiview_detector):

    python demo_infer.py \
        --data_root /path/to/Data \
        --dataset wildtrack \
        --checkpoint /path/to/MultiviewDetector.pth \
        --frame_idx 0 \
        --out_dir demo_out

Cấu trúc thư mục dữ liệu mong đợi (mặc định của repo):
    <data_root>/Wildtrack/...
    <data_root>/MultiviewX/...
Nếu --data_root trỏ THẲNG vào thư mục dataset (chứa sẵn "Image_subsets"), script
cũng tự nhận diện được, không bắt buộc phải có thư mục con Wildtrack/MultiviewX.
"""
import os
import sys
import argparse

import numpy as np

# ---------------------------------------------------------------------------
# Đảm bảo import được gói `multiview_detector` của repo MVDet (đặt demo_infer.py
# cùng cấp với thư mục multiview_detector, hoặc sửa MVDET_ROOT bên dưới).
# ---------------------------------------------------------------------------
MVDET_ROOT = os.path.dirname(os.path.abspath(__file__))
sys.path.insert(0, MVDET_ROOT)

import torch
import torch.nn as nn
import torch.nn.functional as F
import torchvision.transforms as T
from PIL import Image

import matplotlib
matplotlib.use('Agg')  # không cần màn hình, chỉ lưu file ảnh
import matplotlib.pyplot as plt

try:
    from kornia.geometry.transform import warp_perspective
except ImportError as e:
    raise ImportError(
        "Thiếu thư viện 'kornia'. Cài bằng: pip install kornia") from e

from torchvision.models.vgg import vgg11

from multiview_detector.datasets import Wildtrack, MultiviewX, frameDataset
from multiview_detector.models.resnet import resnet18
from multiview_detector.utils.nms import nms
from multiview_detector.utils.image_utils import img_color_denormalize, add_heatmap_to_image


# =============================================================================
# 1) MÔ HÌNH - bản sao PerspTransDetector, chạy trên MỘT thiết bị (CPU/1 GPU)
# =============================================================================
class PerspTransDetectorSingleDevice(nn.Module):
    """Kiến trúc giống hệt multiview_detector/models/persp_trans_detector.py,
    chỉ khác là không ép buộc phải có 2 GPU. Tên các sub-module giữ nguyên
    (base_pt1, base_pt2, img_classifier, world_classifier) nên state_dict của
    checkpoint gốc load thẳng vào được."""

    def __init__(self, dataset, arch='resnet18', device='cpu'):
        super().__init__()
        self.device = device
        self.num_cam = dataset.num_cam
        self.img_shape, self.reducedgrid_shape = dataset.img_shape, dataset.reducedgrid_shape

        imgcoord2worldgrid_matrices = self.get_imgcoord2worldgrid_matrices(
            dataset.base.intrinsic_matrices, dataset.base.extrinsic_matrices,
            dataset.base.worldgrid2worldcoord_mat)
        self.coord_map = self.create_coord_map(self.reducedgrid_shape + [1])

        self.upsample_shape = list(map(lambda x: int(x / dataset.img_reduce), self.img_shape))
        img_reduce = np.array(self.img_shape) / np.array(self.upsample_shape)
        img_zoom_mat = np.diag(np.append(img_reduce, [1]))
        map_zoom_mat = np.diag(np.append(np.ones([2]) / dataset.grid_reduce, [1]))
        self.proj_mats = [torch.from_numpy(map_zoom_mat @ imgcoord2worldgrid_matrices[cam] @ img_zoom_mat)
                           for cam in range(self.num_cam)]

        if arch == 'vgg11':
            base = vgg11().features
            base[-1] = nn.Sequential()
            base[-4] = nn.Sequential()
            split = 10
            self.base_pt1 = base[:split]
            self.base_pt2 = base[split:]
            out_channel = 512
        elif arch == 'resnet18':
            base = nn.Sequential(*list(
                resnet18(replace_stride_with_dilation=[False, True, True]).children())[:-2])
            split = 7
            self.base_pt1 = base[:split]
            self.base_pt2 = base[split:]
            out_channel = 512
        else:
            raise Exception('architecture currently support [vgg11, resnet18]')

        self.img_classifier = nn.Sequential(
            nn.Conv2d(out_channel, 64, 1), nn.ReLU(),
            nn.Conv2d(64, 2, 1, bias=False))
        # LƯU Ý: một số checkpoint cũ (vd. mốc 2020-12-18 tải từ link README gốc) đặt tên
        # lớp phân loại cuối là "world_classifier" thay vì "map_classifier" như trong file
        # persp_trans_detector.py hiện tại trên nhánh master. Dùng đúng tên "world_classifier"
        # ở đây để checkpoint nạp đúng vào layer, tránh bị load_state_dict bỏ qua âm thầm.
        self.world_classifier = nn.Sequential(
            nn.Conv2d(out_channel * self.num_cam + 2, 512, 3, padding=1), nn.ReLU(),
            nn.Conv2d(512, 512, 3, padding=2, dilation=2), nn.ReLU(),
            nn.Conv2d(512, 1, 3, padding=4, dilation=4, bias=False))

        self.to(device)

    def forward(self, imgs, visualize=False):
        B, N, C, H, W = imgs.shape
        assert N == self.num_cam
        world_features = []
        imgs_result = []
        for cam in range(self.num_cam):
            img_feature = self.base_pt1(imgs[:, cam].to(self.device))
            img_feature = self.base_pt2(img_feature)
            img_feature = F.interpolate(img_feature, self.upsample_shape, mode='bilinear')
            img_res = self.img_classifier(img_feature)
            imgs_result.append(img_res)

            proj_mat = self.proj_mats[cam].repeat([B, 1, 1]).float().to(self.device)
            world_feature = warp_perspective(img_feature, proj_mat, self.reducedgrid_shape)
            world_features.append(world_feature)

        world_features = torch.cat(
            world_features + [self.coord_map.repeat([B, 1, 1, 1]).to(self.device)], dim=1)
        map_result = self.world_classifier(world_features)
        map_result = F.interpolate(map_result, self.reducedgrid_shape, mode='bilinear')
        return map_result, imgs_result

    def get_imgcoord2worldgrid_matrices(self, intrinsic_matrices, extrinsic_matrices,
                                         worldgrid2worldcoord_mat):
        projection_matrices = {}
        for cam in range(self.num_cam):
            worldcoord2imgcoord_mat = intrinsic_matrices[cam] @ np.delete(extrinsic_matrices[cam], 2, 1)
            worldgrid2imgcoord_mat = worldcoord2imgcoord_mat @ worldgrid2worldcoord_mat
            imgcoord2worldgrid_mat = np.linalg.inv(worldgrid2imgcoord_mat)
            permutation_mat = np.array([[0, 1, 0], [1, 0, 0], [0, 0, 1]])
            projection_matrices[cam] = permutation_mat @ imgcoord2worldgrid_mat
        return projection_matrices

    def create_coord_map(self, img_size, with_r=False):
        H, W, C = img_size
        grid_x, grid_y = np.meshgrid(np.arange(W), np.arange(H))
        grid_x = torch.from_numpy(grid_x / (W - 1) * 2 - 1).float()
        grid_y = torch.from_numpy(grid_y / (H - 1) * 2 - 1).float()
        ret = torch.stack([grid_x, grid_y], dim=0).unsqueeze(0)
        if with_r:
            rr = torch.sqrt(torch.pow(grid_x, 2) + torch.pow(grid_y, 2)).view([1, 1, H, W])
            ret = torch.cat([ret, rr], dim=1)
        return ret


# =============================================================================
# 2) TIỆN ÍCH
# =============================================================================
def resolve_dataset_root(data_root, dataset_name):
    """Tìm thư mục dataset thật sự, hỗ trợ cả 2 kiểu:
       - data_root/Wildtrack (hoặc MultiviewX)
       - data_root chính là thư mục dataset (đã chứa Image_subsets)
    """
    canonical = 'Wildtrack' if dataset_name == 'wildtrack' else 'MultiviewX'
    candidate_sub = os.path.join(data_root, canonical)
    if os.path.isdir(os.path.join(candidate_sub, 'Image_subsets')):
        return candidate_sub
    if os.path.isdir(os.path.join(data_root, 'Image_subsets')):
        return data_root
    raise FileNotFoundError(
        f"Không tìm thấy thư mục dataset hợp lệ cho '{dataset_name}'. "
        f"Đã thử: '{candidate_sub}' và '{data_root}'. "
        f"Hãy kiểm tra lại --data_root (thư mục phải chứa 'Image_subsets').")


def build_model_and_load_ckpt(test_set, arch, checkpoint_path, device):
    model = PerspTransDetectorSingleDevice(test_set, arch=arch, device=device)
    state_dict = torch.load(checkpoint_path, map_location=device)
    # một số checkpoint được lưu dạng {'state_dict': ...} hoặc {'model': ...}
    if isinstance(state_dict, dict) and 'state_dict' in state_dict:
        state_dict = state_dict['state_dict']
    elif isinstance(state_dict, dict) and 'model' in state_dict:
        state_dict = state_dict['model']

    # Tương thích ngược: một số checkpoint cũ (vd. bản 2020-12-18 tải từ link README gốc)
    # đặt tên lớp phân loại cuối là "world_classifier" thay vì "map_classifier" (tên dùng
    # trên nhánh master hiện tại). Tự động remap để checkpoint nào cũng nạp đúng layer,
    # tránh bị load_state_dict(strict=False) âm thầm bỏ qua.
    own_keys = set(model.state_dict().keys())
    remapped = {}
    for k, v in state_dict.items():
        if k in own_keys:
            remapped[k] = v
        elif k.startswith('world_classifier.') and k.replace('world_classifier.', 'map_classifier.') in own_keys:
            remapped[k.replace('world_classifier.', 'map_classifier.')] = v
        elif k.startswith('map_classifier.') and k.replace('map_classifier.', 'world_classifier.') in own_keys:
            remapped[k.replace('map_classifier.', 'world_classifier.')] = v
        else:
            remapped[k] = v  # giữ nguyên, sẽ hiện trong "unexpected" nếu vẫn không khớp
    state_dict = remapped
    missing, unexpected = model.load_state_dict(state_dict, strict=False)
    if missing or unexpected:
        # phân loại: num_batches_tracked (BatchNorm) thường vô hại, khác version PyTorch
        harmless = lambda k: k.endswith('num_batches_tracked')
        missing_real = [k for k in missing if not harmless(k)]
        unexpected_real = [k for k in unexpected if not harmless(k)]

        print(f"[Cảnh báo] load_state_dict không khớp hoàn toàn.")
        print(f"  - thiếu (missing): {len(missing)} tensor, trong đó {len(missing_real)} KHÔNG phải "
              f"num_batches_tracked")
        for k in missing:
            print(f"      missing: {k}")
        print(f"  - dư (unexpected): {len(unexpected)} tensor, trong đó {len(unexpected_real)} KHÔNG phải "
              f"num_batches_tracked")
        for k in unexpected:
            print(f"      unexpected: {k}")

        if missing_real or unexpected_real:
            print("  >> CÓ key quan trọng (không phải num_batches_tracked) bị lệch — đây rất có thể là "
                  "nguyên nhân model dự đoán sai/không ra kết quả. Kiểm tra lại --arch có đúng với "
                  "checkpoint đã train không (resnet18 vs vgg11).")
        else:
            print("  >> Toàn bộ key lệch chỉ là 'num_batches_tracked' (BatchNorm) — CHỈ khác version "
                  "PyTorch lúc train/lúc load, KHÔNG ảnh hưởng tới kết quả suy luận. An toàn để bỏ qua.")
    model.eval()
    return model


def detections_from_heatmap(map_res, base, grid_reduce, cls_thres, nms_dist_thres=20):
    """map_res: tensor [1,1,Hr,Wr] -> danh sách vị trí (world grid đầy đủ, world coord mét)."""
    map_grid_res = map_res.detach().cpu().squeeze()
    v_s = map_grid_res[map_grid_res > cls_thres].unsqueeze(1)
    grid_ij = (map_grid_res > cls_thres).nonzero()
    if grid_ij.numel() == 0:
        return np.empty([0, 2]), np.empty([0, 2]), np.empty([0])

    if base.indexing == 'xy':
        grid_xy = grid_ij[:, [1, 0]]
    else:
        grid_xy = grid_ij

    positions = grid_xy.float() * grid_reduce  # về lại full-resolution grid
    scores = v_s.squeeze(1)
    ids, count = nms(positions, scores, nms_dist_thres, np.inf)
    kept_grid = positions[ids[:count]].numpy()
    kept_scores = scores[ids[:count]].numpy()

    world_coords = np.stack(
        [base.get_worldcoord_from_worldgrid(g) for g in kept_grid], axis=0
    ) if len(kept_grid) else np.empty([0, 2])

    return kept_grid, world_coords, kept_scores


def gt_positions_from_map(map_gt, base, grid_reduce):
    """Trích toạ độ ground-truth (để vẽ so sánh) từ map_gt [1,1,Hr,Wr]."""
    m = map_gt.detach().cpu().squeeze()
    grid_ij = (m > 0).nonzero()
    if grid_ij.numel() == 0:
        return np.empty([0, 2])
    if base.indexing == 'xy':
        grid_xy = grid_ij[:, [1, 0]]
    else:
        grid_xy = grid_ij
    positions = (grid_xy.float() * grid_reduce).numpy()
    return positions


def quick_precision_recall(map_res, map_gt, cls_thres):
    pred = (map_res > cls_thres).int().to(map_gt.device)
    tp = (pred.eq(map_gt.int()) * pred.eq(1)).sum().item()
    fp = pred.sum().item() - tp
    fn = map_gt.sum().item() - tp
    precision = tp / (tp + fp + 1e-4)
    recall = tp / (tp + fn + 1e-4)
    return precision, recall


# =============================================================================
# 3) TRỰC QUAN HOÁ
# =============================================================================
def save_camera_montage(imgs_denorm, out_path, max_cols=4):
    n = imgs_denorm.shape[0]
    cols = min(max_cols, n)
    rows = int(np.ceil(n / cols))
    fig, axes = plt.subplots(rows, cols, figsize=(4 * cols, 2.4 * rows))
    axes = np.array(axes).reshape(-1)
    for cam in range(n):
        img = imgs_denorm[cam].permute(1, 2, 0).clamp(0, 1).numpy()
        axes[cam].imshow(img)
        axes[cam].set_title(f'Camera {cam + 1}', fontsize=10)
        axes[cam].axis('off')
    for ax in axes[n:]:
        ax.axis('off')
    fig.suptitle('Ảnh đầu vào từ các camera')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_cam_heatmap(imgs_denorm, imgs_res, cam, out_path):
    img = imgs_denorm[cam].permute(1, 2, 0).clamp(0, 1).numpy()
    img_pil = Image.fromarray((img * 255).astype('uint8'))
    heatmap_foot = imgs_res[cam][0, 1].detach().cpu().numpy()
    result = add_heatmap_to_image(heatmap_foot, img_pil)
    result.save(out_path)


def save_bev_comparison(pred_grid, gt_grid, worldgrid_shape, out_path, title_extra=''):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5.5))
    H, W = worldgrid_shape  # N_row, N_col
    for ax, pts, title, color in zip(
            axes, [gt_grid, pred_grid], ['Ground truth', f'Dự đoán (sau NMS){title_extra}'],
            ['tab:blue', 'tab:red']):
        ax.set_facecolor('#f2f2f2')
        if len(pts):
            ax.scatter(pts[:, 1], pts[:, 0], s=18, c=color, edgecolors='k', linewidths=0.3)
        ax.set_xlim(0, H)
        ax.set_ylim(W, 0)
        ax.set_title(f'{title}  (n={len(pts)})')
        ax.set_aspect('equal')
        ax.set_xlabel('grid theo trục ngang')
        ax.set_ylabel('grid theo trục dọc')
    fig.suptitle('Bản đồ mặt đất (bird-eye view) - vị trí người đi bộ')
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


def save_occupancy_heatmaps(map_res, map_gt, out_path):
    fig, axes = plt.subplots(1, 2, figsize=(11, 5))
    im0 = axes[0].imshow(map_res.detach().cpu().squeeze().numpy(), cmap='jet')
    axes[0].set_title('Occupancy map - dự đoán (điểm số thô)')
    fig.colorbar(im0, ax=axes[0], fraction=0.046)
    im1 = axes[1].imshow(map_gt.detach().cpu().squeeze().numpy(), cmap='jet')
    axes[1].set_title('Occupancy map - ground truth')
    fig.colorbar(im1, ax=axes[1], fraction=0.046)
    fig.tight_layout()
    fig.savefig(out_path, dpi=130)
    plt.close(fig)


# =============================================================================
# 4) MAIN
# =============================================================================
def main():
    parser = argparse.ArgumentParser(description='Demo suy luận cho MVDet')
    parser.add_argument('--data_root', type=str, required=True,
                         help='Thư mục chứa dataset (chứa Wildtrack/ và MultiviewX/, '
                              'hoặc trỏ thẳng vào 1 thư mục dataset)')
    parser.add_argument('-d', '--dataset', type=str, default='wildtrack',
                         choices=['wildtrack', 'multiviewx'])
    parser.add_argument('--checkpoint', type=str, required=True,
                         help='Đường dẫn tới file checkpoint .pth (MultiviewDetector.pth)')
    parser.add_argument('--arch', type=str, default='resnet18', choices=['vgg11', 'resnet18'])
    parser.add_argument('--frame_idx', type=int, default=0,
                         help='Chỉ số frame trong tập TEST để demo (0 = frame test đầu tiên)')
    parser.add_argument('--num_frames', type=int, default=1,
                         help='Số frame liên tiếp muốn chạy demo (>=1), dùng để tính precision/recall trung bình')
    parser.add_argument('--cls_thres', type=float, default=0.4, help='Ngưỡng phân loại occupancy map')
    parser.add_argument('--nms_dist_thres', type=float, default=20,
                         help='Ngưỡng khoảng cách (đơn vị: ô lưới full-res) cho NMS')
    parser.add_argument('--device', type=str, default='auto', choices=['auto', 'cpu', 'cuda'])
    parser.add_argument('--out_dir', type=str, default='demo_out')
    args = parser.parse_args()

    device = ('cuda' if torch.cuda.is_available() else 'cpu') if args.device == 'auto' else args.device
    if device == 'cuda' and not torch.cuda.is_available():
        print('[Cảnh báo] Không tìm thấy GPU, chuyển sang chạy CPU (sẽ chậm hơn).')
        device = 'cpu'
    print(f'>> Thiết bị sử dụng: {device}')

    os.makedirs(args.out_dir, exist_ok=True)

    # ---- 1) Dataset ----
    dataset_dir = resolve_dataset_root(args.data_root, args.dataset)
    print(f'>> Thư mục dataset: {dataset_dir}')
    base = Wildtrack(dataset_dir) if args.dataset == 'wildtrack' else MultiviewX(dataset_dir)

    normalize = T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    denormalize = img_color_denormalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225))
    test_trans = T.Compose([T.Resize([720, 1280]), T.ToTensor(), normalize])

    print('>> Đang chuẩn bị tập test (lần đầu có thể mất một lúc để build ground-truth)...')
    test_set = frameDataset(base, train=False, transform=test_trans, grid_reduce=4)
    print(f'>> Tập test có {len(test_set)} frame, {test_set.num_cam} camera / frame.')

    # ---- 2) Model ----
    print(f'>> Đang khởi tạo mô hình PerspTransDetector (arch={args.arch}) và nạp checkpoint...')
    model = build_model_and_load_ckpt(test_set, args.arch, args.checkpoint, device)

    # ---- 3) Chạy demo trên (các) frame ----
    n_frames = max(1, args.num_frames)
    precisions, recalls, det_counts = [], [], []

    for k in range(n_frames):
        idx = args.frame_idx + k
        if idx >= len(test_set):
            print(f'[Bỏ qua] frame_idx={idx} vượt quá số frame test ({len(test_set)}).')
            continue

        imgs, map_gt, imgs_gt, frame_no = test_set[idx]
        imgs_b = imgs.unsqueeze(0).to(device)          # [1, N, C, H, W]
        map_gt_b = map_gt.unsqueeze(0)                  # [1, 1, Hr, Wr]

        with torch.no_grad():
            map_res, imgs_res = model(imgs_b)

        mr = map_res.detach().cpu()
        print(f'   [debug] map_res: min={mr.min().item():.4f}  max={mr.max().item():.4f}  '
              f'mean={mr.mean().item():.4f}  (ngưỡng cls_thres={args.cls_thres})')
        if mr.max().item() < args.cls_thres:
            print('   [debug] >> Giá trị lớn nhất của map_res VẪN THẤP HƠN ngưỡng cls_thres '
                  '=> KHÔNG có ô lưới nào vượt ngưỡng => 0 detection là hệ quả trực tiếp của điều này, '
                  'không phải lỗi ở bước NMS.')

        precision, recall = quick_precision_recall(map_res.cpu(), map_gt_b, args.cls_thres)
        pred_grid, world_coords, scores = detections_from_heatmap(
            map_res, base, test_set.grid_reduce, args.cls_thres, args.nms_dist_thres)
        gt_grid = gt_positions_from_map(map_gt_b, base, test_set.grid_reduce)

        precisions.append(precision)
        recalls.append(recall)
        det_counts.append(len(pred_grid))

        print(f'\n=== Frame test #{idx} (frame gốc = {int(frame_no)}) ===')
        print(f'   Số người phát hiện được (sau NMS): {len(pred_grid)}  |  Ground truth: {len(gt_grid)}')
        print(f'   Precision (theo ô lưới, ngưỡng={args.cls_thres}): {precision * 100:.1f}%'
              f'   Recall: {recall * 100:.1f}%')
        if len(world_coords):
            print('   Toạ độ mặt đất (mét) của các phát hiện đầu tiên:')
            for i, (wc, sc) in enumerate(zip(world_coords[:10], scores[:10])):
                print(f'     #{i + 1}: x={wc[0]:.2f} m, y={wc[1]:.2f} m, score={sc:.3f}')
            if len(world_coords) > 10:
                print(f'     ... và {len(world_coords) - 10} vị trí khác.')

        # ---- Trực quan hoá cho frame đầu tiên (để tránh sinh quá nhiều file) ----
        if k == 0:
            imgs_denorm = denormalize(imgs.unsqueeze(0)).squeeze(0).clamp(0, 1)

            cam_montage_path = os.path.join(args.out_dir, f'frame{idx}_cameras.jpg')
            save_camera_montage(imgs_denorm, cam_montage_path)

            heatmap_cam0_path = os.path.join(args.out_dir, f'frame{idx}_cam1_foot_heatmap.jpg')
            save_cam_heatmap(imgs_denorm, imgs_res, 0, heatmap_cam0_path)

            occ_path = os.path.join(args.out_dir, f'frame{idx}_occupancy_map.jpg')
            save_occupancy_heatmaps(map_res, map_gt_b, occ_path)

            bev_path = os.path.join(args.out_dir, f'frame{idx}_bev_detections.jpg')
            save_bev_comparison(pred_grid, gt_grid, test_set.worldgrid_shape, bev_path)

            det_txt_path = os.path.join(args.out_dir, f'frame{idx}_detections.txt')
            with open(det_txt_path, 'w') as f:
                f.write('# x_meter y_meter score\n')
                for wc, sc in zip(world_coords, scores):
                    f.write(f'{wc[0]:.3f} {wc[1]:.3f} {sc:.3f}\n')

            print(f'\n>> Đã lưu ảnh minh hoạ vào thư mục: {os.path.abspath(args.out_dir)}')
            print(f'     - {os.path.basename(cam_montage_path)}       : ảnh gốc từ {test_set.num_cam} camera')
            print(f'     - {os.path.basename(heatmap_cam0_path)}: heatmap "chân người" chồng lên camera 1')
            print(f'     - {os.path.basename(occ_path)}   : occupancy map dự đoán vs. ground-truth')
            print(f'     - {os.path.basename(bev_path)}  : vị trí người trên bản đồ mặt đất (bird-eye view)')
            print(f'     - {os.path.basename(det_txt_path)}    : danh sách toạ độ (mét) các phát hiện')

    if len(precisions) > 1:
        print(f'\n=== Trung bình trên {len(precisions)} frame ===')
        print(f'   Precision trung bình: {np.mean(precisions) * 100:.1f}%')
        print(f'   Recall trung bình:    {np.mean(recalls) * 100:.1f}%')
        print(f'   Số phát hiện trung bình / frame: {np.mean(det_counts):.1f}')

    print('\nLưu ý: các chỉ số Precision/Recall ở trên tính nhanh theo ô lưới (không cần MATLAB), '
          'CHỈ mang tính tham khảo cho demo. Muốn có MODA/MODP chính thức như trong bài báo, '
          'hãy dùng multiview_detector/evaluation/ (yêu cầu MATLAB) như README gốc của MVDet hướng dẫn.')


if __name__ == '__main__':
    main()