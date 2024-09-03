import inspect
import os
from collections import OrderedDict
import torch

import psutil

def debug(*args, **kwargs):
    frame_info = inspect.getframeinfo(inspect.currentframe().f_back)
    path = frame_info.filename
    filename = os.path.basename(path)
    lineno = frame_info.lineno

    print(f"[DEBUG][{filename}:{lineno}]", *args, **kwargs)

def show_memory(device: torch.device, prompt=""):
    frame_info = inspect.getframeinfo(inspect.currentframe().f_back)
    path = frame_info.filename
    filename = os.path.basename(path)
    lineno = frame_info.lineno

    dev = str(device)
    if "cuda" in dev:
        ma = torch.cuda.memory_allocated()
        mma = torch.cuda.max_memory_allocated()
        mr = torch.cuda.memory_reserved()
        mmr = torch.cuda.max_memory_reserved()
        info = f"ma: {ma / 2**20:.2f}M, mma: {mma / 2**20:.2f}M, mr: {mr / 2**20:.2f}M, mmr: {mmr / 2**20:.2f}M"
    elif "mps" in dev:
        alloc_mem = torch.mps.current_allocated_memory()
        driver_mem = torch.mps.driver_allocated_memory()
        info = f"alloc_mem: {alloc_mem / 2**20:.2f}M, driver_mem: {driver_mem / 2**20:.2f}M"
    elif "cpu" in dev:
        p = psutil.Process(os.getpid())
        mem_info = p.memory_info()
        mem_percent = p.memory_percent()
        info = f"RSS: {mem_info.rss / 2**20:.2f}M ({mem_percent:.2f}%)"
    else:
        # not implemented
        raise NotImplementedError(f"Unsupported device: {dev}")

    print(f"[MEM-{os.getpid()}-{device}][{filename}:{lineno}] {info} ({prompt})")


def get_model_params_num(model: torch.nn.Module) -> int:
    return sum(p.numel() for p in model.parameters() if p.requires_grad)


# Ref: https://gist.github.com/davesteele/44793cd0348f59f8fadd49d7799bd306
class CacheDict(OrderedDict):
    """Dict with a limited length, ejecting LRUs as needed."""

    def __init__(self, *args, cache_len: int = 10, **kwargs):
        assert cache_len > 0
        self.cache_len = cache_len

        super().__init__(*args, **kwargs)

    def __setitem__(self, key, value):
        super().__setitem__(key, value)
        super().move_to_end(key)

        while len(self) > self.cache_len:
            oldkey = next(iter(self))
            super().__delitem__(oldkey)

    def __getitem__(self, key):
        val = super().__getitem__(key)
        super().move_to_end(key)

        return val
