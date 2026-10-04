import os
import json
from operator import itemgetter

import torch
import numpy as np
from torchvision.datasets import VisionDataset
import torchvision.transforms.functional as F
from PIL import Image

from utils import geom, basic, vox
from utils.projection import image_points_to_world as basic_image_points_to_world


class PedestrianDataset(VisionDataset):
    def __init__(
            self,
            base,
            is_train=True,
            resolution=(160, 4, 250),
            bounds=(-500, 500, -320, 320, 0, 2),
        final_dim: tuple = (720, 1280),
        resize_lim: list = (0.8, 1.2),
        drop_ratio=0,
        pseudo_cache=None,
        pseudo_conf_threshold=0.2,
        pseudo_sigma_m=0.5,
        pseudo_suppress_radius_m=1.0,
        pseudo_fuse_radius_m=0.5,
        pseudo_min_views=1,
    ):
        super().__init__(base.root)
        self.base = base
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        # img_shape and worldgrid_shape is the original shape matching the annotations in dataset
        # MultiviewX: [1080, 1920], [640, 1000] Wildtrack: [1080, 1920], [480, 1440]
        self.img_shape = base.img_shape
        self.worldgrid_shape = base.worldgrid_shape
        self.is_train = is_train
        self.bounds = bounds
        self.resolution = resolution
        self.data_aug_conf = {'final_dim': final_dim, 'resize_lim': resize_lim}
        self.kernel_size = 1.5
        self.max_objects = 60
        self.img_downsample = 4
        self.drop_ratio = int(drop_ratio)
        self.pseudo_cache_path = pseudo_cache if self.is_train else None
        self.pseudo_conf_threshold = float(pseudo_conf_threshold)
        self.pseudo_sigma_m = float(pseudo_sigma_m)
        self.pseudo_suppress_radius_m = float(pseudo_suppress_radius_m)
        self.pseudo_fuse_radius_m = float(pseudo_fuse_radius_m)
        self.pseudo_min_views = int(pseudo_min_views)
        self.pseudo_detections = None
        if self.pseudo_cache_path:
            with open(self.pseudo_cache_path, 'r', encoding='utf-8') as cache_file:
                pseudo_cache_data = json.load(cache_file)
            cache_dataset = pseudo_cache_data.get('dataset')
            if cache_dataset and cache_dataset.lower() != base.__name__.lower():
                raise ValueError(f'Pseudo cache is for {cache_dataset}, but dataset is {base.__name__}')
            self.pseudo_detections = pseudo_cache_data.get('detections', {})
            if not isinstance(self.pseudo_detections, dict):
                raise ValueError("Pseudo cache must contain a 'detections' object keyed by frame and camera")

        # Train: optional dropped labels. Val/test: always full GT for fair metrics.
        if self.drop_ratio > 0 and self.is_train:
            self.anno_dir = os.path.join(
                self.root, 'drop_annotations', f'drop_{self.drop_ratio}', 'annotations_positions'
            )
        else:
            self.anno_dir = os.path.join(self.root, 'annotations_positions')
        if not os.path.isdir(self.anno_dir):
            raise FileNotFoundError(f'Annotation directory not found: {self.anno_dir}')

        self.Y, self.Z, self.X = self.resolution
        self.scene_centroid = torch.tensor((0., 0., 0.)).reshape([1, 3])

        self.vox_util = vox.VoxelUtil(
            self.Y, self.Z, self.X,
            scene_centroid=self.scene_centroid,
            bounds=self.bounds,
            assert_cube=False)

        if self.is_train:
            frame_range = range(0, int(self.num_frame * 0.9))
        else:
            frame_range = range(int(self.num_frame * 0.9), self.num_frame)

        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.world_gt = {}
        self.imgs_gt = {}
        self.pid_dict = {}
        self.download(frame_range)

        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        self.prepare_gt()

        self.calibration = {}
        self.setup()

    def setup(self):
        intrinsic = torch.tensor(np.stack(self.base.intrinsic_matrices, axis=0), dtype=torch.float32)  # S,3,3
        intrinsic = geom.merge_intrinsics(*geom.split_intrinsics(intrinsic)).squeeze()  # S,4,4
        self.calibration['intrinsic'] = intrinsic
        self.calibration['extrinsic'] = torch.eye(4)[None].repeat(intrinsic.shape[0], 1, 1)
        self.calibration['extrinsic'][:, :3] = torch.tensor(
            np.stack(self.base.extrinsic_matrices, axis=0), dtype=torch.float32)

    def prepare_gt(self):
        og_gt = []
        full_anno_dir = os.path.join(self.root, 'annotations_positions')
        for fname in sorted(os.listdir(full_anno_dir)):
            if not fname.endswith('.json'):
                continue
            frame = int(fname.split('.')[0])
            with open(os.path.join(full_anno_dir, fname)) as json_file:
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
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def download(self, frame_range):
        num_frame, num_world_bbox, num_imgs_bbox = 0, 0, 0
        for fname in sorted(os.listdir(self.anno_dir)):
            if not fname.endswith('.json'):
                continue
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                num_frame += 1
                with open(os.path.join(self.anno_dir, fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                world_pts, world_pids = [], []
                img_bboxs, img_pids = [[] for _ in range(self.num_cam)], [[] for _ in range(self.num_cam)]

                for pedestrian in all_pedestrians:
                    grid_x, grid_y = self.base.get_worldgrid_from_pos(pedestrian['positionID']).squeeze()
                    if pedestrian['personID'] not in self.pid_dict:
                        self.pid_dict[pedestrian['personID']] = len(self.pid_dict)
                    num_world_bbox += 1
                    world_pts.append((grid_x, grid_y))
                    world_pids.append(pedestrian['personID'])
                    for cam in range(self.num_cam):
                        if itemgetter('xmin', 'ymin', 'xmax', 'ymax')(pedestrian['views'][cam]) != (-1, -1, -1, -1):
                            img_bboxs[cam].append(itemgetter('xmin', 'ymin', 'xmax', 'ymax')
                                                  (pedestrian['views'][cam]))
                            img_pids[cam].append(pedestrian['personID'])
                            num_imgs_bbox += 1
                # Keep tensor shapes even when a frame/camera has 0 boxes after drop.
                self.world_gt[frame] = (
                    torch.as_tensor(world_pts, dtype=torch.float32).reshape(-1, 2),
                    torch.as_tensor(world_pids, dtype=torch.float32).reshape(-1),
                )
                self.imgs_gt[frame] = {}
                for cam in range(self.num_cam):
                    # x1y1x2y2
                    self.imgs_gt[frame][cam] = (
                        torch.as_tensor(img_bboxs[cam], dtype=torch.float32).reshape(-1, 4),
                        torch.as_tensor(img_pids[cam], dtype=torch.float32).reshape(-1),
                    )

    def get_bev_gt(self, mem_pts, mem_pts_prev, pids, pids_pre):
        center = torch.zeros((1, self.Y, self.X), dtype=torch.float32)
        valid_mask = torch.zeros((1, self.Y, self.X), dtype=torch.bool)
        offset = torch.zeros((4, self.Y, self.X), dtype=torch.float32)
        person_ids = torch.zeros((1, self.Y, self.X), dtype=torch.long)

        prev_pts = dict(zip(pids_pre.int().tolist(), mem_pts_prev[0]))

        for pts, pid in zip(mem_pts[0], pids):
            ct = pts[:2]
            ct_int = ct.int()

            if ct_int[0] < 0 or ct_int[0] >= self.X or ct_int[1] < 0 or ct_int[1] >= self.Y:
                continue

            for c in center:
                basic.draw_umich_gaussian(c, ct_int, self.kernel_size)
            valid_mask[:, ct_int[1], ct_int[0]] = 1
            offset[:2, ct_int[1], ct_int[0]] = ct - ct_int
            person_ids[:, ct_int[1], ct_int[0]] = pid

            if pid in pids_pre:
                t_off = prev_pts[pid.int().item()][:2] - ct_int
                if t_off.abs().max() > 15:
                    continue
                offset[2:, ct_int[1], ct_int[0]] = t_off

        return center, valid_mask, person_ids, offset

    def get_img_gt(self, img_pts, img_pids, sx, sy, crop):
        H = int(self.data_aug_conf['final_dim'][0] / self.img_downsample)
        W = int(self.data_aug_conf['final_dim'][1] / self.img_downsample)
        center = torch.zeros((3, H, W), dtype=torch.float32)
        offset = torch.zeros((2, H, W), dtype=torch.float32)
        size = torch.zeros((2, H, W), dtype=torch.float32)
        valid_mask = torch.zeros((1, H, W), dtype=torch.bool)
        person_ids = torch.zeros((1, H, W), dtype=torch.long)

        xmin = (img_pts[:, 0] * sx - crop[0]) / self.img_downsample
        ymin = (img_pts[:, 1] * sy - crop[1]) / self.img_downsample
        xmax = (img_pts[:, 2] * sx - crop[0]) / self.img_downsample
        ymax = (img_pts[:, 3] * sy - crop[1]) / self.img_downsample

        center_pts = np.stack(((xmin + xmax) / 2, (ymin + ymax) / 2), axis=1)
        center_pts = torch.tensor(center_pts, dtype=torch.float32)
        size_pts = np.stack(((-xmin + xmax), (-ymin + ymax)), axis=1)
        size_pts = torch.tensor(size_pts, dtype=torch.float32)
        foot_pts = np.stack(((xmin + xmax) / 2, ymin), axis=1)
        foot_pts = torch.tensor(foot_pts, dtype=torch.float32)
        head_pts = np.stack(((xmin + xmax) / 2, ymax), axis=1)
        head_pts = torch.tensor(head_pts, dtype=torch.float32)

        for pt_idx, (pid, wh) in enumerate(zip(img_pids, size_pts)):
            for idx, pt in enumerate((foot_pts[pt_idx], )):  # , center_pts[pt_idx], head_pts[pt_idx])):
                if pt[0] < 0 or pt[0] >= W or pt[1] < 0 or pt[1] >= H:
                    continue
                basic.draw_umich_gaussian(center[idx], pt.int(), self.kernel_size)

            ct_int = foot_pts[pt_idx].int()
            if ct_int[0] < 0 or ct_int[0] >= W or ct_int[1] < 0 or ct_int[1] >= H:
                continue
            valid_mask[:, ct_int[1], ct_int[0]] = 1
            offset[:, ct_int[1], ct_int[0]] = foot_pts[pt_idx] - ct_int
            size[:, ct_int[1], ct_int[0]] = wh
            person_ids[:, ct_int[1], ct_int[0]] = pid

        return center, offset, size, person_ids, valid_mask

    def sample_augmentation(self):
        fH, fW = self.data_aug_conf['final_dim']
        if self.is_train:
            resize = np.random.uniform(*self.data_aug_conf['resize_lim'])
            resize_dims = (int(fW * resize), int(fH * resize))
            newW, newH = resize_dims

            # center it
            crop_h = int((newH - fH) / 2)
            crop_w = int((newW - fW) / 2)

            crop_offset = int(self.data_aug_conf['resize_lim'][0] * self.data_aug_conf['final_dim'][0])
            crop_w = crop_w + int(np.random.uniform(-crop_offset, crop_offset))
            crop_h = crop_h + int(np.random.uniform(-crop_offset, crop_offset))

            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        else:  # validation/test
            # do a perfect resize
            resize_dims = (fW, fH)
            crop_h = 0
            crop_w = 0
            crop = (crop_w, crop_h, crop_w + fW, crop_h + fH)
        return resize_dims, crop

    def get_image_data(self, frame, cameras):
        imgs, intrins, extrins = [], [], []
        centers, offsets, sizes, pids, valids = [], [], [], [], []
        for cam in cameras:
            img = Image.open(self.img_fpaths[cam][frame]).convert('RGB')
            W, H = img.size

            resize_dims, crop = self.sample_augmentation()
            sx = resize_dims[0] / float(W)
            sy = resize_dims[1] / float(H)

            extrin = self.calibration['extrinsic'][cam]
            intrin = self.calibration['intrinsic'][cam]
            intrin = geom.scale_intrinsics(intrin.unsqueeze(0), sx, sy).squeeze(0)

            fx, fy, x0, y0 = geom.split_intrinsics(intrin.unsqueeze(0))

            new_x0 = x0 - crop[0]
            new_y0 = y0 - crop[1]

            pix_T_cam = geom.merge_intrinsics(fx, fy, new_x0, new_y0)
            intrin = pix_T_cam.squeeze(0)  # 4,4
            img = basic.img_transform(img, resize_dims, crop)

            imgs.append(F.to_tensor(img))
            intrins.append(intrin)
            extrins.append(extrin)

            img_pts, img_pids = self.imgs_gt[frame][cam]
            center_img, offset_img, size_img, pid_img, valid_img = self.get_img_gt(img_pts, img_pids, sx, sy, crop)

            centers.append(center_img)
            offsets.append(offset_img)
            sizes.append(size_img)
            pids.append(pid_img)
            valids.append(valid_img)

        return torch.stack(imgs), torch.stack(intrins), torch.stack(extrins), torch.stack(centers), torch.stack(
            offsets), torch.stack(sizes), torch.stack(pids), torch.stack(valids)

    def __len__(self):
        return len(self.world_gt.keys())

    def __getitem__(self, index):
        frame = list(self.world_gt.keys())[index]
        pre_frame = list(self.world_gt.keys())[max(index - 1, 0)]
        cameras = list(range(self.num_cam))

        # images
        imgs, intrins, extrins, centers_img, offsets_img, sizes_img, pids_img, valids_img \
            = self.get_image_data(frame, cameras)

        worldcoord_from_worldgrid = torch.eye(4)
        worldcoord_from_worldgrid2d = torch.tensor(self.base.worldcoord_from_worldgrid_mat, dtype=torch.float32)
        worldcoord_from_worldgrid[:2, :2] = worldcoord_from_worldgrid2d[:2, :2]
        worldcoord_from_worldgrid[:2, 3] = worldcoord_from_worldgrid2d[:2, 2]
        worldgrid_T_worldcoord = torch.inverse(worldcoord_from_worldgrid)
        augment = torch.eye(4)

        worldgrid_pts_org, world_pids = self.world_gt[frame]
        worldgrid_pts_pre, world_pid_pre = self.world_gt[pre_frame]

        worldgrid_pts = torch.cat((worldgrid_pts_org, torch.zeros_like(worldgrid_pts_org[:, 0:1])), dim=1).unsqueeze(0)
        worldgrid_pts_pre = torch.cat((worldgrid_pts_pre, torch.zeros_like(worldgrid_pts_pre[:, 0:1])), dim=1)

        if self.is_train:
            Rz = torch.eye(3)
            scene_center = torch.tensor([0., 0., 0.], dtype=torch.float32)
            off = 0.25
            scene_center[:2].uniform_(-off, off)
            augment = geom.merge_rt(Rz.unsqueeze(0), -scene_center.unsqueeze(0)).squeeze()
            worldgrid_T_worldcoord = torch.matmul(augment, worldgrid_T_worldcoord)
            worldgrid_pts = geom.apply_4x4(augment.unsqueeze(0), worldgrid_pts)

        mem_pts = self.vox_util.Ref2Mem(worldgrid_pts, self.Y, self.Z, self.X)
        mem_pts_pre = self.vox_util.Ref2Mem(worldgrid_pts_pre.unsqueeze(0), self.Y, self.Z, self.X)
        center_bev, valid_bev, pid_bev, offset_bev = self.get_bev_gt(mem_pts, mem_pts_pre,  world_pids, world_pid_pre)

        if self.pseudo_detections is not None:
            pseudo_center_bev, pseudo_weight_bev = self.get_pseudo_bev_target(
                frame, worldgrid_T_worldcoord, mem_pts)

        grid_gt = torch.zeros((self.max_objects, 3), dtype=torch.long)
        grid_gt[:worldgrid_pts.shape[1], :2] = worldgrid_pts_org
        grid_gt[:worldgrid_pts.shape[1], 2] = world_pids

        item = {
            'img': imgs,  # S,3,H,W
            'intrinsic': intrins,  # S,4,4
            'extrinsic': extrins,  # S,4,4
            'ref_T_global': worldgrid_T_worldcoord,  # 4,4
            'frame': frame // self.base.frame_step,
            'sequence_num': int(0),
            'grid_gt': grid_gt,
        }

        target = {
            # bev
            'valid_bev': valid_bev,  # 1,Y,X
            'center_bev': center_bev,  # 1,Y,X
            'offset_bev': offset_bev,  # 2,Y,X
            'pid_bev': pid_bev,  # 1,Y,X
            # img
            'center_img': centers_img,  # S,1,H/8,W/8
            'offset_img': offsets_img,  # S,2,H/8,W/8
            'size_img': sizes_img,  # S,2,H/8,W/8
            'valid_img': valids_img,  # S,1,H/8,W/8
            'pid_img': pids_img  # S,1,H/8,W/8
        }
        if self.pseudo_detections is not None:
            target['pseudo_center_bev'] = pseudo_center_bev
            target['pseudo_weight_bev'] = pseudo_weight_bev

        return item, target

    def get_pseudo_bev_target(self, frame, worldgrid_T_worldcoord, mem_gt_pts):
        """Project cached detections into the augmented BEV voxel grid."""
        pseudo_center = torch.zeros((1, self.Y, self.X), dtype=torch.float32)
        pseudo_weight = torch.zeros_like(pseudo_center)
        frame_detections = self.pseudo_detections.get(str(frame), {})
        if not isinstance(frame_detections, dict):
            return pseudo_center, pseudo_weight

        image_points_by_cam, scores_by_cam = {}, {}
        for camera_key, detections in frame_detections.items():
            try:
                camera = int(camera_key)
            except (TypeError, ValueError):
                continue
            if camera < 0 or camera >= self.num_cam:
                continue
            image_points, scores = [], []
            for detection in detections:
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
                image_points.append(((x1 + x2) * 0.5, y2))
                scores.append(float(np.clip(score, 0.0, 1.0)))
            if image_points:
                image_points_by_cam[camera] = image_points
                scores_by_cam[camera] = scores

        world_points, scores, cameras = [], [], []
        for camera, image_points in image_points_by_cam.items():
            projected = basic_image_points_to_world(
                image_points, self.base.intrinsic_matrices[camera], self.base.extrinsic_matrices[camera])
            valid = np.isfinite(projected).all(axis=1)
            world_points.extend(projected[valid].tolist())
            scores.extend(np.asarray(scores_by_cam[camera], dtype=np.float32)[valid].tolist())
            cameras.extend([camera] * int(valid.sum()))
        if not world_points:
            return pseudo_center, pseudo_weight

        world_xyz = torch.zeros((1, len(world_points), 3), dtype=torch.float32)
        world_xyz[0, :, :2] = torch.as_tensor(world_points, dtype=torch.float32)
        ref_xyz = geom.apply_4x4(worldgrid_T_worldcoord.unsqueeze(0), world_xyz)
        mem_pts = self.vox_util.Ref2Mem(ref_xyz, self.Y, self.Z, self.X)[0]
        gt_pts = mem_gt_pts[0, :, :2]

        worldgrid_mat = np.asarray(self.base.worldcoord_from_worldgrid_mat, dtype=np.float64)
        units_to_m = 0.01 if 'wildtrack' in self.base.__name__.lower() else 1.0
        voxel_size_x_m = ((self.bounds[1] - self.bounds[0]) / self.X *
                          np.linalg.norm(worldgrid_mat[:2, 0]) * units_to_m)
        voxel_size_y_m = ((self.bounds[3] - self.bounds[2]) / self.Y *
                          np.linalg.norm(worldgrid_mat[:2, 1]) * units_to_m)

        finite = torch.isfinite(mem_pts[:, :2]).all(dim=1)
        mem_pts, scores, cameras = mem_pts[finite, :2], torch.as_tensor(scores)[finite], torch.as_tensor(cameras)[finite]
        mem_pts, scores = self.fuse_pseudo_views(mem_pts, scores, cameras, voxel_size_x_m, voxel_size_y_m)
        sigma_x = max(self.pseudo_sigma_m / max(voxel_size_x_m, 1e-6), 0.5)
        sigma_y = max(self.pseudo_sigma_m / max(voxel_size_y_m, 1e-6), 0.5)
        radius_x = max(int(np.ceil(3 * sigma_x)), 1)
        radius_y = max(int(np.ceil(3 * sigma_y)), 1)
        suppress_x = self.pseudo_suppress_radius_m / max(voxel_size_x_m, 1e-6)
        suppress_y = self.pseudo_suppress_radius_m / max(voxel_size_y_m, 1e-6)

        for point, score in zip(mem_pts, scores.tolist()):
            center_x, center_y = point[:2]
            x, y = int(center_x.item()), int(center_y.item())
            if x < 0 or x >= self.X or y < 0 or y >= self.Y:
                continue
            if gt_pts.numel():
                gt_dist_sq = ((gt_pts[:, 0] - center_x) * voxel_size_x_m) ** 2 + \
                             ((gt_pts[:, 1] - center_y) * voxel_size_y_m) ** 2
                if torch.any(gt_dist_sq <= self.pseudo_suppress_radius_m ** 2):
                    continue

            x0, x1 = max(0, x - radius_x), min(self.X, x + radius_x + 1)
            y0, y1 = max(0, y - radius_y), min(self.Y, y + radius_y + 1)
            patch_x = torch.arange(x0, x1, dtype=torch.float32)
            patch_y = torch.arange(y0, y1, dtype=torch.float32)
            yy, xx = torch.meshgrid(patch_y, patch_x, indexing='ij')
            # Centre on the integer cell like get_bev_gt, so the peak is exactly 1 there.
            gaussian = torch.exp(-0.5 * (((xx - x) / sigma_x) ** 2 +
                                         ((yy - y) / sigma_y) ** 2))
            if gt_pts.numel():
                gt_dx = (gt_pts[:, 0, None, None] - xx) * voxel_size_x_m
                gt_dy = (gt_pts[:, 1, None, None] - yy) * voxel_size_y_m
                near_gt = ((gt_dx ** 2 + gt_dy ** 2) <= self.pseudo_suppress_radius_m ** 2).any(dim=0)
                gaussian = gaussian.masked_fill(near_gt, 0.0)
            current_target = pseudo_center[0, y0:y1, x0:x1]
            current_weight = pseudo_weight[0, y0:y1, x0:x1]
            pseudo_center[0, y0:y1, x0:x1] = torch.maximum(current_target, gaussian)
            pseudo_weight[0, y0:y1, x0:x1] = torch.maximum(current_weight, gaussian * score)

        return pseudo_center, pseudo_weight

    def fuse_pseudo_views(self, mem_pts, scores, cameras, voxel_size_x_m, voxel_size_y_m):
        """Merge per-camera projections of the same person into one BEV point.

        MVDet max-merges overlapping per-camera Gaussians and its MSE loss treats the
        result as one soft blob. The focal pseudo loss turns every peak into a hard
        positive, so the cameras are merged explicitly: greedy by score, at most one
        point per camera per cluster, score-weighted mean position, max score.
        Clusters seen by fewer than ``pseudo_min_views`` cameras are dropped.
        """
        if self.pseudo_fuse_radius_m <= 0 or len(mem_pts) == 0:
            keep = torch.ones(len(mem_pts), dtype=torch.bool) if self.pseudo_min_views <= 1 \
                else torch.zeros(len(mem_pts), dtype=torch.bool)
            return mem_pts[keep], scores[keep]

        scale = torch.tensor([voxel_size_x_m, voxel_size_y_m], dtype=torch.float32)
        metric = mem_pts * scale
        dist_sq = ((metric[:, None, :] - metric[None, :, :]) ** 2).sum(dim=-1)
        assigned = torch.zeros(len(mem_pts), dtype=torch.bool)
        fused_pts, fused_scores = [], []
        for seed in torch.argsort(scores, descending=True).tolist():
            if assigned[seed]:
                continue
            members, used_cameras = [seed], {int(cameras[seed])}
            assigned[seed] = True
            near = (dist_sq[seed] <= self.pseudo_fuse_radius_m ** 2) & ~assigned
            for other in torch.nonzero(near).flatten()[torch.argsort(dist_sq[seed][near])].tolist():
                camera = int(cameras[other])
                if camera in used_cameras:
                    continue
                members.append(other)
                used_cameras.add(camera)
                assigned[other] = True
            if len(used_cameras) < self.pseudo_min_views:
                continue
            weights = scores[members]
            fused_pts.append((mem_pts[members] * weights[:, None]).sum(dim=0) / weights.sum())
            fused_scores.append(weights.max())
        if not fused_pts:
            return mem_pts[:0], scores[:0]
        return torch.stack(fused_pts), torch.stack(fused_scores)
