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

        # PATH settings — auto-detect platform by file path (runpod=/workspace, else H cluster).
        platform = 'runpod' if os.path.abspath(__file__).startswith('/workspace') else 'H'
        platform_index = ['runpod', 'H'].index(platform)
        self.sys_home_dir = [
            '/workspace',
            '/mnt/shared-storage-gpfs2/single-cell/zhengpeng',
        ][platform_index]

        # Import sub-configs
        for sub_cfg_cls in (ConfigTask, ConfigData, ConfigModel, ConfigTrain, ConfigLoss, ConfigMisc):
            sub_cfg = sub_cfg_cls(self)
            for k, v in sub_cfg.__dict__.items():
                setattr(self, k, v)


config = Config()
