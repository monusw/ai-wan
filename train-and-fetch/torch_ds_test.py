import os
import time

import torch
from torch.utils.data import Dataset, DataLoader
from torch.utils.data.distributed import DistributedSampler
import torch.distributed as dist

class MyDataset(Dataset):
    def __init__(self, n):
        self.len = n
    def __len__(self):
        return self.len
    def __getitem__(self, index):
        img = torch.randn(3, 224, 224)
        target = torch.randint(0, 1000, (1,))
        return img, target, index


# torchrun --nproc_per_node=2 torch_ds_test.py
# ps -ef
def main():
    world_size = int(os.environ['WORLD_SIZE'])
    rank = int(os.environ['RANK'])
    dist.init_process_group(backend='gloo')

    dataset = MyDataset(102400)
    sampler = DistributedSampler(dataset, shuffle=False, drop_last=True)
    dataloader = DataLoader(dataset, sampler=sampler,
                            batch_size=4,
                            num_workers=2,
                            prefetch_factor=4)
    for epoch in range(2):
        sampler.set_epoch(epoch)
        iterator = iter(dataloader)
        for i, (img, target, index) in enumerate(iterator):
            prefix = f"[RANK {rank}/{world_size}][EPOCH {epoch}][BATCH {i}/{len(dataloader)}]"

            if rank == 0:
                # data queue size, not work on macOS
                # data_queue_size = iterator._data_queue.qsize()
                cache_size = 0
                for worker_id, info in iterator._task_info.items():
                    if len(info) == 2:
                        cache_size += 1
                # print(prefix, f"data_queue_size: {data_queue_size}")
                print(prefix, f"cache_size: {cache_size}")

            print(prefix, index)
            time.sleep(1)


if __name__ == '__main__':
    main()
