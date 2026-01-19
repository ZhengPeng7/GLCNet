

class ConfigLoss:
    def __init__(self, base) -> None:
        # OIM Loss settings (cq_size is defined per-task in task.py)
        self.oim_momentum = 0.5
        self.oim_scalar = 30.0

        # Detection loss weights
        self.lw_rpn_reg = 1
        self.lw_rpn_cls = 1
        self.lw_proposal_reg = 10
        self.lw_proposal_cls = 1
        self.lw_box_reg = 1
        self.lw_box_cls = 1
        self.lw_box_reid = 1
