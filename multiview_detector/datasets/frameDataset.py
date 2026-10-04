import os
import json
import time
from operator import itemgetter
import copy

import numpy as np
from PIL import Image
import kornia
from kornia.geometry.transform import warp_perspective
from torchvision.datasets import VisionDataset
import torch
import torch.nn.functional as F
import torchvision.transforms as T
from multiview_detector.utils.projection import *
from multiview_detector.utils.image_utils import draw_umich_gaussian, random_affine
import matplotlib.pyplot as plt


def get_gt(Rshape, x_s, y_s, w_s=None, h_s=None, v_s=None, reduce=4, top_k=100, kernel_size=4):
    H, W = Rshape
    heatmap = np.zeros([1, H, W], dtype=np.float32)
    reg_mask = np.zeros([top_k], dtype=bool)
    idx = np.zeros([top_k], dtype=np.int64)
    pid = np.zeros([top_k], dtype=np.int64)
    offset = np.zeros([top_k, 2], dtype=np.float32)
    wh = np.zeros([top_k, 2], dtype=np.float32)

    for k in range(len(v_s)):
        ct = np.array([x_s[k] / reduce, y_s[k] / reduce], dtype=np.float32)
        if 0 <= ct[0] < W and 0 <= ct[1] < H:
            ct_int = ct.astype(np.int32)
            draw_umich_gaussian(heatmap[0], ct_int, kernel_size / reduce)
            reg_mask[k] = 1
            idx[k] = ct_int[1] * W + ct_int[0]
            pid[k] = v_s[k]
            offset[k] = ct - ct_int
            if w_s is not None and h_s is not None:
                wh[k] = [w_s[k] / reduce, h_s[k] / reduce]
            # plt.imshow(heatmap[0])
            # plt.show()

    ret = {'heatmap': torch.from_numpy(heatmap), 'reg_mask': torch.from_numpy(reg_mask), 'idx': torch.from_numpy(idx),
           'pid': torch.from_numpy(pid), 'offset': torch.from_numpy(offset)}
    if w_s is not None and h_s is not None:
        ret.update({'wh': torch.from_numpy(wh)})
    return ret


class frameDataset(VisionDataset):
    def __init__(self, base, train=True, reID=False, world_reduce=4, img_reduce=12,
                 world_kernel_size=10, img_kernel_size=10,
                 train_ratio=0.9, top_k=100, force_download=True,
                 semi_supervised=0.0, dropout=0.0, augmentation=False, drop_ratio=None,
                 pseudo_cache=None, pseudo_conf_threshold=0.2, pseudo_sigma_m=0.25,
                 pseudo_suppress_radius_m=1.0):
        super().__init__(base.root)

        self.base = base
        self.num_cam, self.num_frame = base.num_cam, base.num_frame
        # world (grid) reduce: on top of the 2.5cm grid
        self.reID, self.top_k = reID, top_k
        # reduce = input/output
        self.world_reduce, self.img_reduce = world_reduce, img_reduce
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.world_kernel_size, self.img_kernel_size = world_kernel_size, img_kernel_size
        self.semi_supervised = semi_supervised * train
        self.dropout = dropout
        self.drop_ratio = drop_ratio
        self.transform = T.Compose([T.ToTensor(), T.Normalize((0.485, 0.456, 0.406), (0.229, 0.224, 0.225)),
                                    T.Resize((np.array(self.img_shape) * 8 // self.img_reduce).tolist())])
        self.augmentation = augmentation

        self.Rworld_shape = list(map(lambda x: x // self.world_reduce, self.worldgrid_shape))
        self.Rimg_shape = np.ceil(np.array(self.img_shape) / self.img_reduce).astype(int).tolist()

        # offline detector cache (train only): {frame: {cam: [[x1, y1, x2, y2, score], ...]}}
        self.pseudo_cache = None
        self.pseudo_conf_threshold = float(pseudo_conf_threshold)
        self.pseudo_sigma_m = float(pseudo_sigma_m)
        self.pseudo_suppress_radius_m = float(pseudo_suppress_radius_m)
        if pseudo_cache and train:
            with open(pseudo_cache, 'r') as cache_file:
                cache_data = json.load(cache_file)
            cache_dataset = cache_data.get('dataset')
            if cache_dataset and cache_dataset.lower() != base.__name__.lower():
                raise ValueError(f'Pseudo cache is for {cache_dataset}, but dataset is {base.__name__}')
            self.pseudo_cache = cache_data.get('detections', {})
            if not isinstance(self.pseudo_cache, dict):
                raise ValueError("Pseudo cache must contain a 'detections' object keyed by frame and camera")

        # drop_20 -> drop_annotations/drop_20/... ; None -> full annotations
        if drop_ratio is not None:
            self.anno_dir = os.path.join(self.root, 'drop_annotations', drop_ratio, 'annotations_positions')
        else:
            self.anno_dir = os.path.join(self.root, 'annotations_positions')
        if not os.path.isdir(self.anno_dir):
            raise FileNotFoundError(f'Annotation directory not found: {self.anno_dir}')

        if train:
            frame_range = range(0, int(self.num_frame * train_ratio))
        else:
            frame_range = range(int(self.num_frame * train_ratio), self.num_frame)

        self.world_from_img, self.img_from_world = self.get_world_imgs_trans()
        world_masks = torch.ones([self.num_cam, 1] + self.worldgrid_shape)
        self.imgs_region = warp_perspective(world_masks, self.img_from_world, self.img_shape, 'nearest',
                                                   align_corners=False)

        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.world_gt = {}
        self.imgs_gt = {}
        self.pid_dict = {}
        self.keeps = {}
        num_frame, num_world_bbox, num_imgs_bbox = 0, 0, 0
        num_keep, num_all = 0, 0
        for fname in sorted(os.listdir(self.anno_dir)):
            if not fname.endswith('.json'):
                continue
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                num_frame += 1
                keep = np.mean(np.array(frame_range) < frame) < self.semi_supervised if self.semi_supervised else 1
                with open(os.path.join(self.anno_dir, fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                world_pts, world_pids = [], []
                img_bboxs, img_pids = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]
                if keep:
                    for pedestrian in all_pedestrians:
                        grid_x, grid_y = self.base.get_worldgrid_from_pos(pedestrian['positionID']).squeeze()
                        if pedestrian['personID'] not in self.pid_dict:
                            self.pid_dict[pedestrian['personID']] = len(self.pid_dict)
                        num_all += 1
                        num_keep += keep
                        num_world_bbox += keep
                        if self.base.indexing == 'xy':
                            world_pts.append((grid_x, grid_y))
                        else:
                            world_pts.append((grid_y, grid_x))
                        world_pids.append(self.pid_dict[pedestrian['personID']])
                        for cam in range(self.num_cam):
                            if itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]) != (-1, -1, -1, -1):
                                img_bboxs[cam].append(itemgetter('xmin', 'ymin', 'xmax', 'ymax')
                                                      (pedestrian['views'][cam]))
                                img_pids[cam].append(self.pid_dict[pedestrian['personID']])
                                num_imgs_bbox += 1
                self.world_gt[frame] = (
                    np.array(world_pts, dtype=np.float64).reshape(-1, 2),
                    np.array(world_pids, dtype=np.int64).reshape(-1),
                )
                self.imgs_gt[frame] = {}
                for cam in range(self.num_cam):
                    # x1y1x2y2; empty cam -> (0, 4) not 1D (0,)
                    self.imgs_gt[frame][cam] = (
                        np.array(img_bboxs[cam], dtype=np.float64).reshape(-1, 4),
                        np.array(img_pids[cam], dtype=np.int64).reshape(-1),
                    )
                self.keeps[frame] = keep

        keep_ratio = (num_keep / num_all) if num_all else 0.0
        print(f'anno_dir: {self.anno_dir}')
        print(f'all: pid: {len(self.pid_dict)}, frame: {num_frame}, keep ratio: {keep_ratio:.3f}\n'
              f'recorded: world bbox: {num_world_bbox / max(num_frame, 1):.1f}, '
              f'imgs bbox per cam: {num_imgs_bbox / max(num_frame, 1) / self.num_cam:.1f}')
        # gt in mot format for evaluation
        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        if not os.path.exists(self.gt_fpath) or force_download:
            self.prepare_gt()

        pass

    def get_world_imgs_trans(self, z=0):
        # image and world feature maps from xy indexing, change them into world indexing / xy indexing (img)
        # world grid change to xy indexing
        Rworldgrid_from_worldcoord_mat = np.linalg.inv(self.base.worldcoord_from_worldgrid_mat @
                                                       self.base.world_indexing_from_xy_mat)

        # z in meters by default
        # projection matrices: img feat -> world feat
        worldcoord_from_imgcoord_mats = [get_worldcoord_from_imgcoord_mat(self.base.intrinsic_matrices[cam],
                                                                          self.base.extrinsic_matrices[cam],
                                                                          z / self.base.worldcoord_unit)
                                         for cam in range(self.num_cam)]
        # worldgrid(xy)_from_img(xy)
        proj_mats = [Rworldgrid_from_worldcoord_mat @ worldcoord_from_imgcoord_mats[cam] @ self.base.img_xy_from_xy_mat
                     for cam in range(self.num_cam)]
        world_from_img = torch.tensor(np.stack(proj_mats))
        # img(xy)_from_worldgrid(xy)
        img_from_world = torch.tensor(np.stack([np.linalg.inv(proj_mat) for proj_mat in proj_mats]))
        return world_from_img.float(), img_from_world.float()

    def prepare_gt(self):
        og_gt = []
        for fname in sorted(os.listdir(os.path.join(self.root, 'annotations_positions'))):
            frame = int(fname.split('.')[0])
            with open(os.path.join(self.root, 'annotations_positions', fname)) as json_file:
                all_pedestrians = json.load(json_file)
            for single_pedestrian in all_pedestrians:
                def is_in_cam(cam):
                    return not (single_pedestrian['views'][cam]['xmin'] == -1 and
                                single_pedestrian['views'][cam]['xmax'] == -1 and
                                single_pedestrian['views'][cam]['ymin'] == -1 and
                                single_pedestrian['views'][cam]['ymax'] == -1)

                in_cam_range = sum(is_in_cam(cam) for cam in range(self.num_cam))
                if not in_cam_range:
                    continue
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID']).squeeze()
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def __getitem__(self, index, visualize=False):
        def plt_visualize():
            import cv2
            from matplotlib.patches import Circle
            fig, ax = plt.subplots(1)
            ax.imshow(img)
            for i in range(len(img_x_s)):
                x, y = img_x_s[i], img_y_s[i]
                if x > 0 and y > 0:
                    ax.add_patch(Circle((x, y), 10))
            plt.show()
            img0 = img.copy()
            for bbox in img_bboxs:
                bbox = tuple(int(pt) for pt in bbox)
                cv2.rectangle(img0, (bbox[0], bbox[1]), (bbox[2], bbox[3]), (0, 255, 0), 2)
            plt.imshow(img0)
            plt.show()

        frame = list(self.world_gt.keys())[index]
        # imgs
        imgs, imgs_gt, affine_mats, masks = [], [], [], []
        for cam in range(self.num_cam):
            img = np.array(Image.open(self.img_fpaths[cam][frame]).convert('RGB'))
            img_bboxs, img_pids = self.imgs_gt[frame][cam]
            if self.augmentation:
                img, img_bboxs, img_pids, M = random_affine(img, img_bboxs, img_pids)
            else:
                M = np.eye(3)
            imgs.append(self.transform(img))
            affine_mats.append(torch.from_numpy(M).float())
            img_x_s, img_y_s = (img_bboxs[:, 0] + img_bboxs[:, 2]) / 2, img_bboxs[:, 3]
            img_w_s, img_h_s = (img_bboxs[:, 2] - img_bboxs[:, 0]), (img_bboxs[:, 3] - img_bboxs[:, 1])

            img_gt = get_gt(self.Rimg_shape, img_x_s, img_y_s, img_w_s, img_h_s, v_s=img_pids,
                            reduce=self.img_reduce, top_k=self.top_k, kernel_size=self.img_kernel_size)
            imgs_gt.append(img_gt)
            if visualize:
                plt_visualize()

        imgs = torch.stack(imgs)
        affine_mats = torch.stack(affine_mats)
        # inverse_M = torch.inverse(
        #     torch.cat([affine_mats, torch.tensor([0, 0, 1]).view(1, 1, 3).repeat(self.num_cam, 1, 1)], dim=1))[:, :2]
        imgs_gt = {key: torch.stack([img_gt[key] for img_gt in imgs_gt]) for key in imgs_gt[0]}
        # imgs_gt['heatmap_mask'] = self.imgs_region if self.keeps[frame] else torch.zeros_like(self.imgs_region)
        # imgs_gt['heatmap_mask'] = warp_perspective(imgs_gt['heatmap_mask'], affine_mats, self.img_shape,
        #                                                   align_corners=False)
        # imgs_gt['heatmap_mask'] = F.interpolate(imgs_gt['heatmap_mask'], self.Rimg_shape, mode='bilinear',
        #                                         align_corners=False).bool().float()
        drop, keep_cams = np.random.rand() < self.dropout, torch.ones(self.num_cam, dtype=torch.bool)
        if drop:
            drop_cam = np.random.randint(0, self.num_cam)
            keep_cams[drop_cam] = 0
            for key in imgs_gt:
                imgs_gt[key][drop_cam] = 0
        # world gt
        world_pt_s, world_pid_s = self.world_gt[frame]
        world_gt = get_gt(self.Rworld_shape, world_pt_s[:, 0], world_pt_s[:, 1], v_s=world_pid_s,
                          reduce=self.world_reduce, top_k=self.top_k, kernel_size=self.world_kernel_size)
        if self.pseudo_cache is not None:
            world_gt['pseudo_heatmap'], world_gt['pseudo_weight'] = self.get_pseudo_world_gt(frame, world_pt_s)
        return imgs, world_gt, imgs_gt, affine_mats, frame

    def get_pseudo_world_gt(self, frame, world_pt_s):
        """Project cached detector foot points onto the reduced BEV and rasterize soft evidence.

        Cache boxes are in original image coordinates, so image augmentation does not affect them.
        Pseudo points (and blob cells) within ``pseudo_suppress_radius_m`` of a GT point are dropped.
        """
        H, W = self.Rworld_shape
        pseudo_heatmap = np.zeros([1, H, W], dtype=np.float32)
        pseudo_weight = np.zeros([1, H, W], dtype=np.float32)
        frame_detections = self.pseudo_cache.get(str(frame), {})
        if not isinstance(frame_detections, dict) or not frame_detections:
            return torch.from_numpy(pseudo_heatmap), torch.from_numpy(pseudo_weight)

        # world_pt_s is (col, row) on the full grid, same convention as get_gt
        gt_cols = (world_pt_s[:, 0] / self.world_reduce).astype(np.int64)
        gt_rows = (world_pt_s[:, 1] / self.world_reduce).astype(np.int64)
        cell_size_m = 0.025 * self.world_reduce  # both datasets use a 2.5cm world grid
        sigma_cells = max(self.pseudo_sigma_m / cell_size_m, 0.5)
        suppress_cells = self.pseudo_suppress_radius_m / cell_size_m
        radius = max(int(np.ceil(3.0 * sigma_cells)), 1)
        kernel_axis = np.arange(-radius, radius + 1, dtype=np.float32)
        kernel_y, kernel_x = np.meshgrid(kernel_axis, kernel_axis, indexing='ij')
        gaussian = np.exp(-(kernel_x ** 2 + kernel_y ** 2) / (2.0 * sigma_cells ** 2))

        for cam_key, boxes in frame_detections.items():
            try:
                cam = int(cam_key)
            except (TypeError, ValueError):
                continue
            if cam < 0 or cam >= self.num_cam:
                continue
            image_points, scores = [], []
            for detection in boxes:
                if isinstance(detection, dict):
                    box = detection.get('bbox', detection.get('box'))
                    score = float(detection.get('confidence', detection.get('score', 0.0)))
                else:
                    if len(detection) < 5:
                        continue
                    box, score = detection[:4], float(detection[4])
                if box is None or score < self.pseudo_conf_threshold:
                    continue
                x1, y1, x2, y2 = map(float, box)
                if not np.isfinite([x1, y1, x2, y2, score]).all() or x2 <= x1 or y2 <= y1:
                    continue
                image_points.append([(x1 + x2) * 0.5, y2])
                scores.append(float(np.clip(score, 0.0, 1.0)))
            if not image_points:
                continue
            world_coords = get_worldcoord_from_imagecoord(np.asarray(image_points, dtype=np.float64).T,
                                                          self.base.intrinsic_matrices[cam],
                                                          self.base.extrinsic_matrices[cam])
            grid_pts = self.base.get_worldgrid_from_worldcoord(world_coords)
            for i, score in enumerate(scores):
                grid_x, grid_y = grid_pts[:, i]
                if self.base.indexing == 'xy':
                    col, row = int(grid_x // self.world_reduce), int(grid_y // self.world_reduce)
                else:
                    col, row = int(grid_y // self.world_reduce), int(grid_x // self.world_reduce)
                if not (0 <= row < H and 0 <= col < W):
                    continue
                if gt_rows.size and np.min((gt_rows - row) ** 2 + (gt_cols - col) ** 2) <= suppress_cells ** 2:
                    continue
                y0, y1 = max(0, row - radius), min(H, row + radius + 1)
                x0, x1 = max(0, col - radius), min(W, col + radius + 1)
                ky0, kx0 = y0 - (row - radius), x0 - (col - radius)
                patch = gaussian[ky0:ky0 + (y1 - y0), kx0:kx0 + (x1 - x0)]
                if gt_rows.size:
                    patch_rows, patch_cols = np.ogrid[y0:y1, x0:x1]
                    near_gt = np.min((gt_rows[:, None, None] - patch_rows) ** 2 +
                                     (gt_cols[:, None, None] - patch_cols) ** 2, axis=0) <= suppress_cells ** 2
                    patch = np.where(near_gt, 0.0, patch)
                np.maximum(pseudo_heatmap[0, y0:y1, x0:x1], patch, out=pseudo_heatmap[0, y0:y1, x0:x1])
                np.maximum(pseudo_weight[0, y0:y1, x0:x1], patch * score, out=pseudo_weight[0, y0:y1, x0:x1])

        return torch.from_numpy(pseudo_heatmap), torch.from_numpy(pseudo_weight)

    def __len__(self):
        return len(self.world_gt.keys())


def test(test_projection=False):
    from torch.utils.data import DataLoader
    from multiview_detector.datasets.Wildtrack import Wildtrack
    from multiview_detector.datasets.MultiviewX import MultiviewX

    dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), train=True, augmentation=False)
    # dataset = frameDataset(MultiviewX(os.path.expanduser('~/Data/MultiviewX')), train=True)
    # dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), train=True, semi_supervised=.1)
    # dataset = frameDataset(MultiviewX(os.path.expanduser('~/Data/MultiviewX')), train=True, semi_supervised=.1)
    # dataset = frameDataset(Wildtrack(os.path.expanduser('~/Data/Wildtrack')), train=True, semi_supervised=0.5)
    # dataset = frameDataset(MultiviewX(os.path.expanduser('~/Data/MultiviewX')), train=True, semi_supervised=0.5)
    min_dist = np.inf
    for world_gt in dataset.world_gt.values():
        x, y = world_gt[0][:, 0], world_gt[0][:, 1]
        if x.size and y.size:
            xy_dists = ((x - x[:, None]) ** 2 + (y - y[:, None]) ** 2) ** 0.5
            np.fill_diagonal(xy_dists, np.inf)
            min_dist = min(min_dist, np.min(xy_dists))
            pass
    dataloader = DataLoader(dataset, 2, True, num_workers=0)
    # imgs, world_gt, imgs_gt, M, frame = next(iter(dataloader))
    t0 = time.time()
    imgs, world_gt, imgs_gt, M, frame = dataset.__getitem__(0, visualize=False)
    print(time.time() - t0)

    pass
    if test_projection:
        import matplotlib.pyplot as plt
        from multiview_detector.utils.projection import get_worldcoord_from_imagecoord
        world_grid_maps = []
        xx, yy = np.meshgrid(np.arange(0, 1920, 20), np.arange(0, 1080, 20))
        H, W = xx.shape
        image_coords = np.stack([xx, yy], axis=2).reshape([-1, 2])
        for cam in range(dataset.num_cam):
            world_coords = get_worldcoord_from_imagecoord(image_coords.transpose(),
                                                          dataset.base.intrinsic_matrices[cam],
                                                          dataset.base.extrinsic_matrices[cam])
            world_grids = dataset.base.get_worldgrid_from_worldcoord(world_coords).transpose().reshape([H, W, 2])
            world_grid_map = np.zeros(dataset.worldgrid_shape)
            for i in range(H):
                for j in range(W):
                    x, y = world_grids[i, j]
                    if dataset.base.indexing == 'xy':
                        if x in range(dataset.worldgrid_shape[1]) and y in range(dataset.worldgrid_shape[0]):
                            world_grid_map[int(y), int(x)] += 1
                    else:
                        if x in range(dataset.worldgrid_shape[0]) and y in range(dataset.worldgrid_shape[1]):
                            world_grid_map[int(x), int(y)] += 1
            world_grid_map = world_grid_map != 0
            plt.imshow(world_grid_map)
            plt.show()
            world_grid_maps.append(world_grid_map)
            pass
        plt.imshow(np.sum(np.stack(world_grid_maps), axis=0))
        plt.show()
        pass


if __name__ == '__main__':
    test(True)
