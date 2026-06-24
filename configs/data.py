

class ConfigData:
    def __init__(self, base) -> None:
        # Input image settings (min_size, max_size are defined per-task in task.py)

        # Sentinel person-id assigned to unlabeled people across all datasets
        # (CUHK-SYSU / PRW / MovieNet-PS). Boxes with this pid are detected but
        # excluded from labeled-identity matching and OIM's labeled lookup table.
        self.unlabeled_pid = 5555

        # Batch size (batch_size_train is the single source of truth in configs/config.py)
        self.batch_size_test = 1

        # DataLoader settings
        self.num_workers_train = min(base.batch_size_train, 8) if hasattr(base, 'batch_size_train') else 3
        self.num_workers_test = 4  # Prefetch data while GPU is processing
