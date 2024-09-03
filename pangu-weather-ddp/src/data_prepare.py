import os
import glob
import numpy as np
from datetime import datetime, timedelta
from torch.utils.data import DataLoader

from era5_data.utils_data import compute_and_save_data_stats
from era5_data.dataset import Era5Dataset
from era5_data.config import cfg


def get_begin_end_date(data_path):
    path = os.path.join(data_path, 'surface', '*.nc')
    begin_date = None
    end_date = None
    for filename in glob.glob(path):
        filename = os.path.basename(filename)
        date = filename.split('.')[0]
        if begin_date is None:
            begin_date = date
            end_date = date
        else:
            if date < begin_date:
                begin_date = date
            if date > end_date:
                end_date = date
    end_ts = datetime.strptime(end_date, '%Y%m%d')
    end_ts += timedelta(days=1)
    end_date = end_ts.strftime('%Y%m%d')
    return begin_date, end_date


def main():
    begin_date, end_date = get_begin_end_date(cfg.PG_INPUT_PATH)
    # All data
    dataset = Era5Dataset(cfg.PG_INPUT_PATH,
                          begin_date=begin_date,
                          end_date=end_date,
                          freq='1h',
                          lead_time=0,
                          normalize_data_path=None)
    print(f"begin_date: {begin_date}, end_date: {end_date}, total data points: {len(dataset)}")
    loader = DataLoader(dataset, batch_size=1, shuffle=False)
    data_dir = os.path.join(cfg.PG_INPUT_PATH, 'ext_data')
    data_path = os.path.join(data_dir, 'data_statistics.npz')
    os.makedirs(data_dir, exist_ok=True)
    # 1. compute and save data statistics (mean, std)
    compute_and_save_data_stats(loader, outpath=data_path)


if __name__ == "__main__":
    main()

