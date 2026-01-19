

class ConfigData:
    def __init__(self, base) -> None:
        # Input image settings (min_size, max_size are defined per-task in task.py)

        # Batch size settings
        self.batch_size_train = 3
        self.batch_size_test = 1

        # DataLoader settings
        self.num_workers_train = min(base.batch_size_train, 8) if hasattr(base, 'batch_size_train') else 3
        self.num_workers_test = 4  # Prefetch data while GPU is processing
