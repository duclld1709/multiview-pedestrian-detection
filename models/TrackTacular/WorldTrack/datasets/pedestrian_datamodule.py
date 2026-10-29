import os
from typing import Optional

import lightning as pl
from torch.utils.data import DataLoader

from datasets.multiviewx_dataset import MultiviewX
from datasets.wildtrack_dataset import Wildtrack
from datasets.pedestrian_dataset import PedestrianDataset
from datasets.sampler import TemporalSampler


class PedestrianDataModule(pl.LightningDataModule):
    def __init__(
            self,
            data_dir: str = "../data/MultiviewX",
            batch_size: int = 2,
            num_workers: int = 4,
            resolution=None,
            bounds=None,
            accumulate_grad_batches=8,
            drop_ratio: int = 0,
            drop_mode: str = 'instance',
            pseudo_cache: Optional[str] = None,
            pseudo_conf_threshold: float = 0.5,
            pseudo_sigma_m: float = 0.3,
            pseudo_suppress_radius_m: float = 1.0,
    ):
        super().__init__()
        self.data_dir = data_dir
        self.batch_size = batch_size
        self.num_workers = num_workers
        self.resolution = resolution
        self.bounds = bounds
        self.accumulate_grad_batches = accumulate_grad_batches
        # 0 = full labels; 20/45/60 = drop_annotations/drop_XX (train split only)
        assert drop_ratio in (0, 20, 45, 60), f'drop_ratio must be 0, 20, 45 or 60, got {drop_ratio}'
        self.drop_ratio = drop_ratio
        # instance = drop_XX (people dropped per frame); frame = frame_drop_XX (whole train frames dropped)
        assert drop_mode in ('instance', 'frame'), f"drop_mode must be 'instance' or 'frame', got {drop_mode}"
        self.drop_mode = drop_mode
        # offline detector pseudo labels, train split only (needs model.use_pseudo_labels true)
        self.pseudo_cache = pseudo_cache
        self.pseudo_conf_threshold = pseudo_conf_threshold
        self.pseudo_sigma_m = pseudo_sigma_m  # 2x the 0.15 m GT kernel
        self.pseudo_suppress_radius_m = pseudo_suppress_radius_m
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

        if stage == 'fit':
            self.data_train = PedestrianDataset(
                base,
                is_train=True,
                resolution=self.resolution,
                bounds=self.bounds,
                drop_ratio=self.drop_ratio,
                drop_mode=self.drop_mode,
                pseudo_cache=self.pseudo_cache,
                pseudo_conf_threshold=self.pseudo_conf_threshold,
                pseudo_sigma_m=self.pseudo_sigma_m,
                pseudo_suppress_radius_m=self.pseudo_suppress_radius_m,
            )
        if stage == 'fit' or stage == 'validate':
            self.data_val = PedestrianDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds,
            )
        if stage == 'test':
            self.data_test = PedestrianDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds
            )
        if stage == 'predict':
            self.data_predict = PedestrianDataset(
                base,
                is_train=False,
                resolution=self.resolution,
                bounds=self.bounds,
            )

    def train_dataloader(self):
        return DataLoader(
            self.data_train,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=TemporalSampler(self.data_train, batch_size=self.batch_size,
                                    accumulate_grad_batches=self.accumulate_grad_batches),
            pin_memory=True,
        )

    def val_dataloader(self):
        return DataLoader(
            self.data_val,
            batch_size=self.batch_size,
            num_workers=self.num_workers,
            sampler=TemporalSampler(self.data_val, batch_size=self.batch_size,
                                    accumulate_grad_batches=self.accumulate_grad_batches),
            pin_memory=True,
        )

    def test_dataloader(self):
        return DataLoader(
            self.data_test,
            batch_size=1,
            num_workers=1,
        )

    def predict_dataloader(self):
        return DataLoader(
            self.data_predict,
            batch_size=self.batch_size,
            num_workers=self.num_workers
        )
