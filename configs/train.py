import math


class ConfigTrain:
    def __init__(self, base) -> None:
        # Faster-training settings
        self.precisionHigh = True
        self.compile = False
        self.mixed_precision = ['bf16', 'fp16', 'no'][0]

        # Optimizer selection: SGD, AdamW, Lion
        self.optimizer = ['SGD', 'AdamW', 'Lion'][2]

        # Learning rate settings
        self.lr = 3e-3
        self.lr_scaled = self.lr * (base.batch_size_train / 3)
        self.clip_gradients = 10.0

        # SGD-specific settings (used when optimizer='SGD')
        self.sgd_momentum = 0.9
        self.sgd_weight_decay = 0.0005
        # lr_decay_milestones is defined per-task in task.py
        self.lr_decay_rate = 0.1
        self.warmup_factor = 1.0 / 1000

        # AdamW/Lion settings (used when optimizer='AdamW' or 'Lion')
        self.adamw_weight_decay = 0.05
        self.lion_weight_decay = 0.01
        self.lion_use_triton = True
        self.warmup_epochs = 2
        self.cosine_eta_min = 1e-6
