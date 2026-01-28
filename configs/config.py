import os

from .task import ConfigTask
from .data import ConfigData
from .model import ConfigModel
from .train import ConfigTrain
from .loss import ConfigLoss
from .misc import ConfigMisc


class Config:
    def __init__(self) -> None:
        # Main active settings
        self.batch_size_train = 3

        # PATH settings
        self.sys_home_dir = ['/mnt/shared-storage-gpfs2/single-cell/zhengpeng', '/workspace'][1]

        # Import sub-configs
        for sub_cfg_cls in (ConfigTask, ConfigData, ConfigModel, ConfigTrain, ConfigLoss, ConfigMisc):
            sub_cfg = sub_cfg_cls(self)
            for k, v in sub_cfg.__dict__.items():
                setattr(self, k, v)


config = Config()
