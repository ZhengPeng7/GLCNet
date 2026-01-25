import os


class ConfigTask:
    def __init__(self, base) -> None:
        # Task and dataset selection
        self.task = ['CUHK-SYSU', 'PRW', 'MVN'][1]

        # Dataset-specific settings
        self.dataset_settings = {
            'CUHK-SYSU': {
                'data_dir': 'cuhk_sysu',
                'lut_size': 5532,
                'cq_size': 5000,
                'max_epochs': 15,
                'multistep_milestones': [8],
            },
            'PRW': {
                'data_dir': 'prw',
                'lut_size': 482,
                'cq_size': 500,
                'max_epochs': 8,
                'multistep_milestones': [5],
            },
            'MVN': {
                'data_dir': 'MovieNet-PS',
                'lut_size': 3087,
                'cq_size': 3000,
                'max_epochs': 30,
                'multistep_milestones': [20],
                'min_size': 900,
                'max_size': 1500,
            },
        }

        # MovieNet-PS specific settings
        self.mvn_train_appN = [10, 30, 50, 70, 100][0]
        self.mvn_gallery_size = [2000, 4000, 10000][0]

        # Get current task settings
        task_cfg = self.dataset_settings[self.task]
        self.lut_size = task_cfg['lut_size']
        self.cq_size = task_cfg['cq_size']
        self.max_epochs = task_cfg['max_epochs']
        self.multistep_milestones = task_cfg['multistep_milestones']
        self.input_min_size = task_cfg.get('min_size', 900)
        self.input_max_size = task_cfg.get('max_size', 1500)
        self.output_dir = 'ckpts/tmp'  # Default, overridden by --ckpt_dir

        # Data paths
        self.data_root = os.path.join(base.sys_home_dir, 'datasets/ps', task_cfg['data_dir'])
