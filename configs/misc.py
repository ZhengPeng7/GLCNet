

class ConfigMisc:
    def __init__(self, base) -> None:
        # Device settings
        self.device = ['cuda', 'cpu'][0]

        # Reproducibility
        self.seed = 1

        # Evaluation settings
        self.eval_period = 1
        self.eval_use_gt = False
        self.eval_use_cbgm = False

        # Logging settings
        self.disp_period = 500
        self.tf_board = True
