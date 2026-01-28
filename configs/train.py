class ConfigTrain:
    def __init__(self, base) -> None:
        # Faster-training settings
        self.precisionHigh = True
        self.mixed_precision = ['bf16', 'fp16', 'no'][0]

        # Optimizer selection: SGD, AdamW, Lion
        self.optimizer = ['SGD', 'AdamW', 'Lion'][0]

        # Learning rate settings
        self.lr = 3e-3 * (base.batch_size_train / 3)
        self.clip_gradients = 10.0

        # Warmup settings
        self.warmup_enabled = True
        self.warmup_epochs = 1

        # LR scheduler selection: MultiStepLR or CosineAnnealingLR
        self.scheduler = ['MultiStepLR', 'CosineAnnealingLR'][0]
        # MultiStepLR settings (multistep_milestones is defined per-task in task.py)
        self.multistep_gamma = 0.1
        # CosineAnnealingLR settings
        self.cosine_eta_min = 1e-6

        # SGD-specific settings (used when optimizer='SGD')
        self.sgd_momentum = 0.9
        self.sgd_weight_decay = 0.0005
        self.sgd_nesterov = True

        # AdamW/Lion settings (used when optimizer='AdamW' or 'Lion')
        self.adamw_weight_decay = 0.05
        self.lion_weight_decay = 0.01
