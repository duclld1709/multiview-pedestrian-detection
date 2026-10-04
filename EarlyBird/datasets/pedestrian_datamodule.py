import lightning as pl
import os
from torch.utils.data import DataLoader
from typing import Optional
from datasets.multiviewx_dataset import MultiviewX
from datasets.sampler import RandomPairSampler
from datasets.wildtrack_dataset import Wildtrack
from datasets.pedestrian_dataset import PedestrianDataset


class PedestrianDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "../data/MultiviewX",
            batch_size: int = 1,
            num_workers: int = 4,
            resolution=None,
            bounds=None,
            load_depth=False,
            drop_ratio: int = 0,
            pseudo_cache: Optional[str] = None,
            pseudo_conf_threshold: float = 0.2,
            pseudo_sigma_m: float = 0.5,
            pseudo_suppress_radius_m: float = 1.0,
            pseudo_fuse_radius_m: float = 0.5,
            pseudo_min_views: int = 1,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.bounds = bounds
        self.load_depth = load_depth
        self.drop_ratio = int(drop_ratio)
        self.pseudo_cache = pseudo_cache
        self.pseudo_conf_threshold = float(pseudo_conf_threshold)
        self.pseudo_sigma_m = float(pseudo_sigma_m)
        self.pseudo_suppress_radius_m = float(pseudo_suppress_radius_m)
        self.pseudo_fuse_radius_m = float(pseudo_fuse_radius_m)
        self.pseudo_min_views = int(pseudo_min_views)
        self.dataset = os.path.basename(self.data_dir)

        self.data_predict = None
        self.data_test = None
        self.data_val = None
        self.data_train = None

    def setup(self, stage: Optional[str] = None):
        if 'wildtrack' in self.dataset.lower():
            base = Wildtrack(self.data_dir)
        elif 'multiviewx' in self.dataset.lower():
            base = MultiviewX(self.data_dir)
        else:
            raise ValueError(f'Unknown dataset name {self.dataset}')

        common = dict(
            resolution=self.resolution,
            bounds=self.bounds,
            drop_ratio=self.drop_ratio,
        )

        if stage == 'fit':
            self.data_train = PedestrianDataset(
                base,
                is_train=True,
                pseudo_cache=self.pseudo_cache,
                pseudo_conf_threshold=self.pseudo_conf_threshold,
                pseudo_sigma_m=self.pseudo_sigma_m,
                pseudo_suppress_radius_m=self.pseudo_suppress_radius_m,
                pseudo_fuse_radius_m=self.pseudo_fuse_radius_m,
                pseudo_min_views=self.pseudo_min_views,
                **common,
            )
        if stage == 'fit' or stage == 'validate':
            self.data_val = PedestrianDataset(
                base,
                is_train=False,
                **common,
            )
        if stage == 'test':
            self.data_test = PedestrianDataset(
                base,
                is_train=False,
                **common,
            )
        if stage == 'predict':
            self.data_predict = PedestrianDataset(
                base,
                is_train=False,
                **common,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
            sampler=RandomPairSampler(self.data_train)
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
