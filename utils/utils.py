import datetime
import errno
import json
import os
import os.path as osp
import pickle
import random
import time
from collections import defaultdict, deque

import numpy as np
import torch
from tabulate import tabulate


# -------------------------------------------------------- #
#                          Logger                          #
# -------------------------------------------------------- #
class SmoothedValue(object):
    """
    Track a series of values and provide access to smoothed values over a
    window or the global series average.
    """

    def __init__(self, window_size=20, fmt=None):
        if fmt is None:
            fmt = "{median:.2f}"
        self.deque = deque(maxlen=window_size)
        self.total = 0.0
        self.count = 0
        self.fmt = fmt

    def update(self, value, n=1):
        self.deque.append(value)
        self.count += n
        self.total += value * n

    @property
    def median(self):
        d = torch.tensor(list(self.deque))
        return d.median().item()

    @property
    def avg(self):
        d = torch.tensor(list(self.deque), dtype=torch.float32)
        return d.mean().item()

    @property
    def global_avg(self):
        return self.total / self.count

    @property
    def max(self):
        return max(self.deque)

    @property
    def value(self):
        return self.deque[-1]

    def __str__(self):
        return self.fmt.format(
            median=self.median,
            avg=self.avg,
            global_avg=self.global_avg,
            max=self.max,
            value=self.value,
        )


class MetricLogger(object):
    def __init__(self, delimiter="\t", log_file="training.log"):
        self.meters = defaultdict(SmoothedValue)
        self.delimiter = delimiter
        self.log_file = log_file
        if self.log_file is not None:
            with open(self.log_file, "w") as f:
                f.write("")

    def update(self, **kwargs):
        for k, v in kwargs.items():
            if isinstance(v, torch.Tensor):
                v = v.item()
            assert isinstance(v, (float, int))
            self.meters[k].update(v)

    def __getattr__(self, attr):
        if attr in self.meters:
            return self.meters[attr]
        if attr in self.__dict__:
            return self.__dict__[attr]
        raise AttributeError("'{}' object has no attribute '{}'".format(type(self).__name__, attr))

    def __str__(self):
        loss_str = []
        for name, meter in self.meters.items():
            loss_str.append("{}: {}".format(name, str(meter)))
        return self.delimiter.join(loss_str)

    def add_meter(self, name, meter):
        self.meters[name] = meter

    def _log_to_console_and_file(self, message):
        if self.log_file is not None:
            with open(self.log_file, "a") as f:
                f.write(message + "\n")

    def log_every(self, iterable, print_freq, header=None):
        i = 0
        if not header:
            header = ""
        start_time = time.time()
        end = time.time()
        iter_time = SmoothedValue(fmt="{avg:.3f}")
        data_time = SmoothedValue(fmt="{avg:.3f}")
        space_fmt = ":" + str(len(str(len(iterable)))) + "d"
        log_msg = self.delimiter.join([header, "[{0" + space_fmt + "}/{1}]", "time: {time_now}", "{meters}"])
        for obj in iterable:
            data_time.update(time.time() - end)
            yield obj
            iter_time.update(time.time() - end)
            if i % print_freq == 0 or i == len(iterable) - 1:
                # time_now = str(datetime.datetime.now())
                time_now = time.asctime(time.gmtime())
                message_to_log = log_msg.format(i, len(iterable), time_now=time_now, meters=str(self))
                self._log_to_console_and_file(message_to_log)
            i += 1
            end = time.time()
        total_time = time.time() - start_time
        total_time_str = str(datetime.timedelta(seconds=int(total_time)))
        final_msg = "{} Total time: {} ({:.4f} s / it)".format(header, total_time_str, total_time / len(iterable))
        self._log_to_console_and_file(final_msg)


# -------------------------------------------------------- #
#                      File operation                      #
# -------------------------------------------------------- #
def filename(path):
    return osp.splitext(osp.basename(path))[0]


def mkdir(path):
    try:
        os.makedirs(path)
    except OSError as e:
        if e.errno != errno.EEXIST:
            raise


def read_json(fpath):
    with open(fpath, "r") as f:
        obj = json.load(f)
    return obj


def write_json(obj, fpath):
    mkdir(osp.dirname(fpath))
    _obj = obj.copy()
    for k, v in _obj.items():
        if isinstance(v, np.ndarray):
            _obj.pop(k)
    with open(fpath, "w") as f:
        json.dump(_obj, f, indent=4, separators=(",", ": "))


def symlink(src, dst, overwrite=True, **kwargs):
    if os.path.lexists(dst) and overwrite:
        os.remove(dst)
    os.symlink(src, dst, **kwargs)


# -------------------------------------------------------- #
#                           Misc                           #
# -------------------------------------------------------- #
def create_small_table(small_dict):
    """
    Create a small table using the keys of small_dict as headers. This is only
    suitable for small dictionaries.

    Args:
        small_dict (dict): a result dictionary of only a few items.

    Returns:
        str: the table as a string.
    """
    keys, values = tuple(zip(*small_dict.items()))
    table = tabulate(
        [values],
        headers=keys,
        tablefmt="pipe",
        floatfmt=".3f",
        stralign="center",
        numalign="center",
    )
    return table


def warmup_lr_scheduler(optimizer, warmup_iters, warmup_factor):
    def f(x):
        if x >= warmup_iters:
            return 1
        alpha = float(x) / warmup_iters
        return warmup_factor * (1 - alpha) + alpha

    return torch.optim.lr_scheduler.LambdaLR(optimizer, f)


def load_weights(ckpt_path, model):
    state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
    state_dict = check_state_dict(state_dict)
    model.load_state_dict(state_dict)
    epoch = int(os.path.splitext(os.path.basename(ckpt_path))[0].split('epoch_')[1].split('-')[0])
    return epoch


def resume_from_ckpt(ckpt_path, model, optimizer=None, lr_scheduler=None, only_eval=False):
    if only_eval:
        state_dict = torch.load(ckpt_path, map_location='cpu', weights_only=True)
        state_dict = check_state_dict(state_dict)
        model.load_state_dict(state_dict)
        print(f"loaded checkpoint {ckpt_path}.")
        return 0
    else:
        ckpt = torch.load(ckpt_path)
        my_state_dict = model.state_dict()
        kv_to_be_updated = {k: v for (k, v) in ckpt["model"].items() if 'reid_loss' not in k}
        my_state_dict.update(kv_to_be_updated)
        model.load_state_dict(my_state_dict, strict=False)
        if optimizer is not None:
            optimizer.load_state_dict(ckpt["optimizer"])
        if lr_scheduler is not None:
            lr_scheduler.load_state_dict(ckpt["lr_scheduler"])
        print(f"loaded checkpoint {ckpt_path}.")
        print(f"model was trained for {ckpt['epoch']} epochs.")
        return ckpt["epoch"]


def set_random_seed(seed):
    torch.manual_seed(seed)
    torch.cuda.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    torch.backends.cudnn.benchmark = False
    torch.backends.cudnn.deterministic = True
    random.seed(seed)
    np.random.seed(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)


def check_state_dict(state_dict, unwanted_prefixes=['module.', '_orig_mod.']):
    for k, v in list(state_dict.items()):
        prefix_length = 0
        for unwanted_prefix in unwanted_prefixes:
            if k[prefix_length:].startswith(unwanted_prefix):
                prefix_length += len(unwanted_prefix)
        state_dict[k[prefix_length:]] = state_dict.pop(k)
    return state_dict
