import torch.nn as nn
import torch.nn.functional as F
from pointnet2_utils import PointNetSetAbstractionMsg, PointNetSetAbstraction

from omegaconf import OmegaConf

class get_model(nn.Module):
    def __init__(self, **kwargs):
        super(get_model, self).__init__()

        self.cfg = OmegaConf.create(kwargs)

        output_dim = self.cfg.model.parameters.output_dim
        self.normal_channel = self.cfg.model.parameters.normal_channel
        in_channel = 3 if self.normal_channel else 0

        self.sa1 = PointNetSetAbstractionMsg(128, [0.1, 0.2, 0.4], [8, 16, 64], in_channel,[[32, 32, 64], [64, 64, 128], [64, 96, 128]])
        self.sa2 = PointNetSetAbstractionMsg(32, [0.3, 0.6, 0.9], [16, 32, 64], 320,[[64, 64, 128], [128, 128, 256], [128, 128, 256]])
        self.sa3 = PointNetSetAbstraction(None, None, None, 640 + 3, [256, 512, 1024], True)
        self.fc1 = nn.Linear(1024, 512)
        self.bn1 = nn.BatchNorm1d(512)
        #self.drop1 = nn.Dropout(0.4)
        self.drop1 = nn.Dropout(0.3)
        self.fc2 = nn.Linear(512, 256)
        self.bn2 = nn.BatchNorm1d(256)
        #self.drop2 = nn.Dropout(0.5)
        self.drop2 = nn.Dropout(0.4)
        self.fc3 = nn.Linear(256, output_dim)

    def forward(self, xyz):
        B, _, _ = xyz.shape
        if self.normal_channel:
            norm = xyz[:, 3:, :]
            xyz = xyz[:, :3, :]
        else:
            norm = None
        l1_xyz, l1_points = self.sa1(xyz, norm)
        l2_xyz, l2_points = self.sa2(l1_xyz, l1_points)
        l3_xyz, l3_points = self.sa3(l2_xyz, l2_points)
        x = l3_points.view(B, 1024)
        x = self.drop1(F.relu(self.bn1(self.fc1(x))))
        x = self.drop2(F.relu(self.bn2(self.fc2(x))))
        x = self.fc3(x)

        return x,l3_points