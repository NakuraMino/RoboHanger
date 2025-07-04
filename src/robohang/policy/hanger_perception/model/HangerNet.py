import os
import sys
import importlib

import numpy as np
import torch
import torch.nn as nn
import torch.nn.functional as F
import pytorch_lightning as pl
from typing import Any

import trimesh
from omegaconf import OmegaConf

class get_loss(nn.Module):
    def __init__(self):
        super(get_loss, self).__init__()

    def forward(self, pred, target):

        B, N, C = target.shape
        loss1 = F.l1_loss(pred.reshape((B, N, C)), target, reduction='none')
        #loss1 = F.mse_loss(pred.reshape((B, N, C)), target, reduction='none')
        total_loss = loss1.sum(dim=-1)

        return total_loss # [B, N]


class get_model(pl.LightningModule):
    def __init__(self, **kwargs):
        super(get_model, self).__init__()

        self.cfg = OmegaConf.create(kwargs)
        self.loss_fn = get_loss()
        self.pred_cnt = None
        self.base_dir = self.cfg.path.base_dir
        
        model_path = os.path.join(self.base_dir, "model")
        sys.path.append(model_path)
        model_modl = importlib.import_module(self.cfg.model.name, model_path)
        self.model = model_modl.get_model(**self.cfg)
        self.save_hyperparameters(self.cfg, logger=False)

    def forward(self, xyz : torch.Tensor) -> tuple:
        return self.model(xyz)

    def _shared_eval_step(self, batch) -> tuple:
        points, target = batch

        points = points.transpose(2, 1)
        points = torch.Tensor(points).to(torch.float32) * 10 # normalize
        target = torch.Tensor(target).to(torch.float32) * 10 # normalize

        if self.cfg.misc.hardware.cuda:
            points, target = points.cuda(), target.cuda()

        pred, trans_feat = self(points)
        loss = self.loss_fn(pred, target).sum()

        return pred, target, loss        

    def training_step(self, batch):
        pred_raw, target_raw, loss = self._shared_eval_step(batch)
        pred, target = pred_raw.detach(), target_raw.detach()

        B, N, C = target.size()
        dist = torch.norm((pred.reshape((B, N, C)) - target), dim=-1)
        
        train_err_left = dist[:, 0].mean()
        train_err_right = dist[:, 1].mean()
        train_err_top = dist[:, 2].mean()
        
        metrics = {
            "train/err_left" : train_err_left,
            "train/err_right" : train_err_right,
            "train/err_top" : train_err_top,
            "train/loss" : loss
        }
        self.log_dict(metrics, logger=True, on_epoch=True)

        return loss

    def validation_step(self, batch : tuple):
        pred_raw, target_raw, loss = self._shared_eval_step(batch)
        pred, target = pred_raw.detach(), target_raw.detach()

        B, N, C = target.size()
        dist = torch.norm((pred.reshape((B, N, C)) - target), dim=-1)
        
        valid_err_left = dist[:, 0].mean()
        valid_err_right = dist[:, 1].mean()
        valid_err_top = dist[:, 2].mean()
        
        metrics = {
            "valid/err_left" : valid_err_left,
            "valid/err_right" : valid_err_right,
            "valid/err_top" : valid_err_top,
            "valid/loss" : loss
        }
        self.log_dict(metrics, logger=True, on_epoch=True)

        return metrics

    def test_step(self, batch : tuple):
        pred_raw, target_raw, loss = self._shared_eval_step(batch)
        pred, target = pred_raw.detach(), target_raw.detach()

        B, N, C = target.size()
        dist = torch.norm((pred.reshape((B, N, C)) - target), dim=-1)
        
        test_err_left = dist[:, 0].mean()
        test_err_right = dist[:, 1].mean()
        test_err_top = dist[:, 2].mean()
        
        metrics = {
            "test/err_left" : test_err_left,
            "test/err_right" : test_err_right,
            "test/err_top" : test_err_top,
            "test/loss" : loss
        }
        self.log_dict(metrics, logger=True, on_epoch=True)

        return metrics

    def predict_step(self, batch):
        points = batch

        points = points.transpose(2, 1)
        points = torch.Tensor(points).to(torch.float32) * 10 # normalize

        if self.cfg.misc.hardware.cuda:
            points = points.cuda()

        pred, trans_feat = self(points)        

        self.pred_cnt = 0 if self.pred_cnt is None else self.pred_cnt + 1
        for idx in range(points.shape[0]):
            pcd = points[idx, :, :].permute(1, 0).detach().cpu().numpy().T
            label = pred[idx].detach().cpu().numpy().reshape((-1, 3)) / 10 # denormalize

            save_dir = os.path.join(self.base_dir, "outputs_pred", str(self.pred_cnt), str(idx))
            os.makedirs(save_dir, exist_ok=True)
            trimesh.PointCloud(pcd).export(os.path.join(save_dir, "pcd.ply"))
            np.save(os.path.join(save_dir, "label.npy"), label)

    def on_predict_end(self):
        self.pred_cnt = None

    def configure_optimizers(self):
        if self.cfg.train.optimizer.name == "Adam":
            optimizer = torch.optim.Adam(
                self.parameters(),
                lr=self.cfg.train.optimizer.learning_rate,
                betas=(0.9, 0.999),
                eps=1e-08,
                weight_decay=self.cfg.train.optimizer.decay_rate
            )
            return optimizer
        else:
            optimizer = torch.optim.SGD(
                self.parameters(),
                lr=self.cfg.train.optimizer.learning_rate, 
                momentum=self.cfg.train.optimizer.momentum
            )
            return optimizer