import os
import json
from scipy.stats import multivariate_normal
from PIL import Image
from scipy.sparse import coo_matrix
from torchvision.datasets import VisionDataset
import torch
from torchvision.transforms import ToTensor
from multiview_detector.utils.projection import *


class frameDataset(VisionDataset):
    def __init__(self, base, train=True, transform=ToTensor(), target_transform=ToTensor(),
                 reID=False, grid_reduce=4, img_reduce=4, train_ratio=0.9, force_download=True,
                 drop_ratio=0, pseudo_cache=None, pseudo_conf_threshold=0.2,
                 pseudo_sigma_m=0.5, pseudo_suppress_radius_m=1.0):
        super().__init__(base.root, transform=transform, target_transform=target_transform)

        map_sigma, map_kernel_size = 20 / grid_reduce, 20
        img_sigma, img_kernel_size = 10 / img_reduce, 10
        self.reID, self.grid_reduce, self.img_reduce = reID, grid_reduce, img_reduce

        self.base = base
        self.root, self.num_cam, self.num_frame = base.root, base.num_cam, base.num_frame
        self.img_shape, self.worldgrid_shape = base.img_shape, base.worldgrid_shape  # H,W; N_row,N_col
        self.reducedgrid_shape = list(map(lambda x: int(x / self.grid_reduce), self.worldgrid_shape))
        self.pseudo_cache = None
        self.pseudo_sigma_m = float(pseudo_sigma_m)
        self.pseudo_suppress_radius_m = float(pseudo_suppress_radius_m)
        self.pseudo_conf_threshold = float(pseudo_conf_threshold)
        if pseudo_cache:
            with open(pseudo_cache, 'r') as cache_file:
                cache_data = json.load(cache_file)
            cache_dataset = cache_data.get('dataset')
            if cache_dataset and cache_dataset.lower() != base.__name__.lower():
                raise ValueError(f'Pseudo cache is for {cache_dataset}, but dataset is {base.__name__}')
            self.pseudo_cache = cache_data.get('detections', {})
            if not isinstance(self.pseudo_cache, dict):
                raise ValueError("Pseudo cache must contain a 'detections' object keyed by frame and camera")

        # Images/calib stay under base.root; labels can come from a drop_* folder.
        if drop_ratio > 0:
            drop_tag = int(drop_ratio)  # 20.0 -> drop_20 (not drop_20.0)
            self.anno_dir = os.path.join(
                self.root, 'drop_annotations', f'drop_{drop_tag}', 'annotations_positions'
            )
        else:
            self.anno_dir = os.path.join(self.root, 'annotations_positions')
        if not os.path.isdir(self.anno_dir):
            raise FileNotFoundError(f'Annotation directory not found: {self.anno_dir}')

        if train:
            frame_range = range(0, int(self.num_frame * train_ratio))
        else:
            frame_range = range(int(self.num_frame * train_ratio), self.num_frame)

        self.img_fpaths = self.base.get_image_fpaths(frame_range)
        self.map_gt = {}
        self.imgs_head_foot_gt = {}
        self.download(frame_range)

        self.gt_fpath = os.path.join(self.root, 'gt.txt')
        if not os.path.exists(self.gt_fpath) or force_download:
            self.prepare_gt()

        x, y = np.meshgrid(np.arange(-map_kernel_size, map_kernel_size + 1),
                           np.arange(-map_kernel_size, map_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        map_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * map_sigma)
        map_kernel = map_kernel / map_kernel.max()
        kernel_size = map_kernel.shape[0]
        self.map_kernel = torch.zeros([1, 1, kernel_size, kernel_size], requires_grad=False)
        self.map_kernel[0, 0] = torch.from_numpy(map_kernel)

        x, y = np.meshgrid(np.arange(-img_kernel_size, img_kernel_size + 1),
                           np.arange(-img_kernel_size, img_kernel_size + 1))
        pos = np.stack([x, y], axis=2)
        img_kernel = multivariate_normal.pdf(pos, [0, 0], np.identity(2) * img_sigma)
        img_kernel = img_kernel / img_kernel.max()
        kernel_size = img_kernel.shape[0]
        self.img_kernel = torch.zeros([2, 2, kernel_size, kernel_size], requires_grad=False)
        self.img_kernel[0, 0] = torch.from_numpy(img_kernel)
        self.img_kernel[1, 1] = torch.from_numpy(img_kernel)
        pass

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
                grid_x, grid_y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                og_gt.append(np.array([frame, grid_x, grid_y]))
        og_gt = np.stack(og_gt, axis=0)
        os.makedirs(os.path.dirname(self.gt_fpath), exist_ok=True)
        np.savetxt(self.gt_fpath, og_gt, '%d')

    def download(self, frame_range):
        for fname in sorted(os.listdir(self.anno_dir)):
            if not fname.endswith('.json'):
                continue
            frame = int(fname.split('.')[0])
            if frame in frame_range:
                with open(os.path.join(self.anno_dir, fname)) as json_file:
                    all_pedestrians = json.load(json_file)
                i_s, j_s, v_s = [], [], []
                head_row_cam_s, head_col_cam_s = [[] for _ in range(self.num_cam)], \
                                                 [[] for _ in range(self.num_cam)]
                foot_row_cam_s, foot_col_cam_s, v_cam_s = [[] for _ in range(self.num_cam)], \
                                                          [[] for _ in range(self.num_cam)], \
                                                          [[] for _ in range(self.num_cam)]
                for single_pedestrian in all_pedestrians:
                    x, y = self.base.get_worldgrid_from_pos(single_pedestrian['positionID'])
                    if self.base.indexing == 'xy':
                        i_s.append(int(y / self.grid_reduce))
                        j_s.append(int(x / self.grid_reduce))
                    else:
                        i_s.append(int(x / self.grid_reduce))
                        j_s.append(int(y / self.grid_reduce))
                    v_s.append(single_pedestrian['personID'] + 1 if self.reID else 1)
                    for cam in range(self.num_cam):
                        x = max(min(int((single_pedestrian['views'][cam]['xmin'] +
                                         single_pedestrian['views'][cam]['xmax']) / 2), self.img_shape[1] - 1), 0)
                        y_head = max(single_pedestrian['views'][cam]['ymin'], 0)
                        y_foot = min(single_pedestrian['views'][cam]['ymax'], self.img_shape[0] - 1)
                        if x > 0 and y > 0:
                            head_row_cam_s[cam].append(y_head)
                            head_col_cam_s[cam].append(x)
                            foot_row_cam_s[cam].append(y_foot)
                            foot_col_cam_s[cam].append(x)
                            v_cam_s[cam].append(single_pedestrian['personID'] + 1 if self.reID else 1)
                occupancy_map = coo_matrix((v_s, (i_s, j_s)), shape=self.reducedgrid_shape)
                self.map_gt[frame] = occupancy_map
                self.imgs_head_foot_gt[frame] = {}
                for cam in range(self.num_cam):
                    img_gt_head = coo_matrix((v_cam_s[cam], (head_row_cam_s[cam], head_col_cam_s[cam])),
                                             shape=self.img_shape)
                    img_gt_foot = coo_matrix((v_cam_s[cam], (foot_row_cam_s[cam], foot_col_cam_s[cam])),
                                             shape=self.img_shape)
                    self.imgs_head_foot_gt[frame][cam] = [img_gt_head, img_gt_foot]

    def __getitem__(self, index):
        frame = list(self.map_gt.keys())[index]
        imgs = []
        for cam in range(self.num_cam):
            fpath = self.img_fpaths[cam][frame]
            img = Image.open(fpath).convert('RGB')
            if self.transform is not None:
                img = self.transform(img)
            imgs.append(img)
        imgs = torch.stack(imgs)
        map_gt = self.map_gt[frame].toarray()
        if self.reID:
            map_gt = (map_gt > 0).int()
        if self.target_transform is not None:
            map_gt = self.target_transform(map_gt)
        imgs_gt = []
        for cam in range(self.num_cam):
            img_gt_head = self.imgs_head_foot_gt[frame][cam][0].toarray()
            img_gt_foot = self.imgs_head_foot_gt[frame][cam][1].toarray()
            img_gt = np.stack([img_gt_head, img_gt_foot], axis=2)
            if self.reID:
                img_gt = (img_gt > 0).int()
            if self.target_transform is not None:
                img_gt = self.target_transform(img_gt)
            imgs_gt.append(img_gt.float())
        result = (imgs, map_gt.float(), imgs_gt, frame)
        if self.pseudo_cache is None:
            return result
        pseudo_target, pseudo_weight = self._build_pseudo_targets(frame, map_gt)
        return result + (pseudo_target, pseudo_weight)

    def _build_pseudo_targets(self, frame, map_gt):
        """Project cached person foot points and rasterize soft BEV evidence."""
        height, width = self.reducedgrid_shape
        pseudo_target = np.zeros((height, width), dtype=np.float32)
        pseudo_weight = np.zeros((height, width), dtype=np.float32)
        frame_detections = self.pseudo_cache.get(str(frame), {})
        if not isinstance(frame_detections, dict):
            return torch.from_numpy(pseudo_target).unsqueeze(0), torch.from_numpy(pseudo_weight).unsqueeze(0)

        gt_array = map_gt.detach().cpu().numpy().squeeze()
        gt_rows, gt_cols = np.nonzero(gt_array > 0)
        cell_size_m = 0.1 * self.grid_reduce / 4.0
        sigma_cells = max(self.pseudo_sigma_m / cell_size_m, 0.5)
        suppress_cells = self.pseudo_suppress_radius_m / cell_size_m
        radius = max(int(np.ceil(3.0 * sigma_cells)), 1)
        # Use one deterministic per-frame noise field so every BEV cell has
        # stable jitter across epochs and overlapping detections agree.
        noise = np.random.default_rng(int(frame)).uniform(
            -0.05, 0.05, size=(height, width)).astype(np.float32)
        noisy_target = np.clip(0.95 + noise, 0.0, 1.0)

        for camera_key, boxes in frame_detections.items():
            try:
                camera = int(camera_key)
            except (TypeError, ValueError):
                continue
            if camera < 0 or camera >= self.num_cam:
                continue
            image_points = []
            scores = []
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
            world_points = get_worldcoord_from_imagecoord(
                np.asarray(image_points, dtype=np.float64).T,
                self.base.intrinsic_matrices[camera], self.base.extrinsic_matrices[camera])
            grid_points = self.base.get_worldgrid_from_worldcoord(world_points)
            for point_index, score in enumerate(scores):
                grid_x, grid_y = grid_points[:, point_index]
                if self.base.indexing == 'xy':
                    col, row = int(grid_x / self.grid_reduce), int(grid_y / self.grid_reduce)
                else:
                    row, col = int(grid_x / self.grid_reduce), int(grid_y / self.grid_reduce)
                if not (0 <= row < height and 0 <= col < width):
                    continue
                if gt_rows.size and np.min((gt_rows - row) ** 2 + (gt_cols - col) ** 2) <= suppress_cells ** 2:
                    continue
                y0, y1 = max(0, row - radius), min(height, row + radius + 1)
                x0, x1 = max(0, col - radius), min(width, col + radius + 1)
                patch_rows, patch_cols = np.ogrid[y0:y1, x0:x1]
                disk = ((patch_rows - row) ** 2 + (patch_cols - col) ** 2) <= radius ** 2
                if gt_rows.size:
                    near_gt = np.min((gt_rows[:, None, None] - patch_rows) ** 2 +
                                     (gt_cols[:, None, None] - patch_cols) ** 2, axis=0) <= suppress_cells ** 2
                    disk &= ~near_gt
                target_patch = np.where(disk, noisy_target[y0:y1, x0:x1], 0.0)
                weight_patch = np.where(disk, score, 0.0)
                np.maximum(pseudo_target[y0:y1, x0:x1], target_patch,
                           out=pseudo_target[y0:y1, x0:x1])
                np.maximum(pseudo_weight[y0:y1, x0:x1], weight_patch,
                           out=pseudo_weight[y0:y1, x0:x1])

        return (torch.from_numpy(pseudo_target).unsqueeze(0),
                torch.from_numpy(pseudo_weight).unsqueeze(0))

    def __len__(self):
        return len(self.map_gt.keys())


def test():
    from multiview_detector.datasets.Wildtrack import Wildtrack
    # from multiview_detector.datasets.MultiviewX import MultiviewX
    from multiview_detector.utils.projection import get_worldcoord_from_imagecoord
    dataset = frameDataset(Wildtrack(os.path.expanduser('../Data/Wildtrack')))
    # test projection
    world_grid_maps = []
    xx, yy = np.meshgrid(np.arange(0, 1920, 20), np.arange(0, 1080, 20))
    H, W = xx.shape
    image_coords = np.stack([xx, yy], axis=2).reshape([-1, 2])
    import matplotlib.pyplot as plt
    for cam in range(dataset.num_cam):
        world_coords = get_worldcoord_from_imagecoord(image_coords.transpose(), dataset.base.intrinsic_matrices[cam],
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
    imgs, map_gt, imgs_gt, _ = dataset.__getitem__(0)
    pass


if __name__ == '__main__':
    test()
