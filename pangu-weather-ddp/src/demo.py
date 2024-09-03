import time
from torch.utils.data import DataLoader

from era5_data.dataset import Era5Dataset
from era5_data.config import cfg


def main():
    dataset = Era5Dataset(cfg.PG_INPUT_PATH,
                          begin_date='20070101',
                          end_date='20070103',
                          freq='1h',
                          lead_time=1,
                          normalize_data_path=None,
                          cache_len=0)
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    begin = time.time()
    for i, (input_upper, input_surface, target_upper, target_surface) in enumerate(loader):
        print(f"Batch {i}:")
        print(f"input_upper: {input_upper.shape}")
        print(f"input_surface: {input_surface.shape}")
        print(f"target_upper: {target_upper.shape}")
        print(f"target_surface: {target_surface.shape}")
    end = time.time()
    print(f"Time: {end - begin:.2f}s")

    # print(f"len: {len(dataset)}, hit: {dataset.data_hit}, miss: {dataset.data_miss}")


if __name__ == "__main__":
    main()
