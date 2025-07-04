import os
from typing import Literal

import numpy as np
from torch.utils.data import Dataset, DataLoader, random_split
import pytorch_lightning as pl
from pytorch_lightning.utilities.types import TRAIN_DATALOADERS

import trimesh
from omegaconf import OmegaConf

class HangerNetDataset(Dataset):
    def __init__(self, root_dir : str, stage : Literal['train', 'test', 'predict']):
        self.root_dir = os.path.join(root_dir, stage, "hanger_pcd")
        self.stage = stage

        self.list_of_points = []
        self.list_of_labels = []

        for hanger in os.listdir(self.root_dir):
            hanger_path = os.path.join(self.root_dir, hanger)
            for pose in os.listdir(hanger_path)[:150]:
                point_cloud = os.path.join(hanger, pose, "pcd.ply")
                self.list_of_points.append(point_cloud)
                #if self.stage != 'predict':
                point_label = os.path.join(hanger, pose, "label.npy")
                self.list_of_labels.append(point_label)
            

    def __len__(self):
        return len(self.list_of_points)

    def __getitem__(self, index : int):
        points_path = os.path.join(self.root_dir, self.list_of_points[index])
        points = trimesh.load(points_path).vertices # [N, 3]

        #if self.stage != 'predict':
        labels_path = os.path.join(self.root_dir, self.list_of_labels[index])
        labels = np.load(labels_path) # [3, 3]        
        return points, labels
        #else:
        #    return points
    
class HangerNetDataModule(pl.LightningDataModule):
    def __init__(self, **kwargs) -> None:
        super().__init__()
        self.cfg = OmegaConf.create(kwargs)

    def prepare_data(self):
        # download dataset, called with one process
        pass

    def setup(self, stage: str) -> None:
        # data transformation
        root_dir = os.path.join(self.cfg.path.base_dir, self.cfg.dataset.name)

        full_dtst = HangerNetDataset(root_dir, "train")
        num_datas = len(full_dtst)
        num_train = int(num_datas * 0.8)
        num_valid = num_datas - num_train

        self.train_dtst, self.valid_dtst = random_split(
            full_dtst, [num_train, num_valid]
        )
        self.test_dtst = HangerNetDataset(root_dir, "test")
        self.pred_dtst = HangerNetDataset(root_dir, "predict")

    def train_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset = self.train_dtst,
            batch_size = self.cfg.dataloader.train.batch_size,
            shuffle = self.cfg.dataloader.train.shuffle,
            num_workers = self.cfg.dataloader.train.num_workers)
    
    def val_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset = self.valid_dtst,
            batch_size = self.cfg.dataloader.valid.batch_size,
            shuffle = self.cfg.dataloader.valid.shuffle,
            num_workers = self.cfg.dataloader.valid.num_workers)

    def test_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset = self.test_dtst,
            batch_size = self.cfg.dataloader.test.batch_size,
            shuffle = self.cfg.dataloader.test.shuffle,
            num_workers = self.cfg.dataloader.test.num_workers)
    
    def predict_dataloader(self) -> TRAIN_DATALOADERS:
        return DataLoader(
            dataset = self.pred_dtst,
            batch_size = self.cfg.dataloader.predict.batch_size,
            shuffle = self.cfg.dataloader.predict.shuffle,
            num_workers = self.cfg.dataloader.predict.num_workers)