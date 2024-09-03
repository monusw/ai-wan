import os
from typing import *

import numpy as np
import pandas as pd
import xarray as xr
from torch.utils.data import Dataset

from era5_data.utils_data import load_data_stats, compute_norm_data
from util import debug, CacheDict


class Era5Dataset(Dataset):
    r""" ERA5 dataset (NetCDF format) wrapper for PyTorch DataLoader.
    Args:
        data_path (str): Directory containing the ERA5 NetCDF files.
            The data should be organized in the following structure:
            data_path
            ├── surface
            │   ├── 20070101.nc
            │   └── ...
            └── upper
                ├── 20070101.nc
                └── ...
            Each NetCDF file contains all the data for a single day. (24 hours)
        begin_date (str): Start date of the dataset in the format 'YYYYMMDD'. (e.g. '20070101')
        end_date (str): End date of the dataset in the format 'YYYYMMDD'. (e.g. '20070102')
        freq (str): Frequency of the data. Format: 'nh', n can be 1-24. Default is '1h'.
        lead_time (int): Lead time of the forecast in hours. Default is 1.
        normalize_data_path (str): .npz file containing the mean and standard deviation of the data.
            keys: 'upper_mean', 'upper_std',    (5, 13, 721, 1440)
                  'surface_mean', 'surface_std' (4, 721, 1440)
            If None, we do not normalize the data. Default is None.
        cache_len (int): Number of datasets to cache in memory. Default is 0.
            OS will do page cache, so cache of datsets is only useful for large datasets
            (reduce xarray parsing time).

    Note: `begin_date` is inclusive, `end_date` is exclusive. For example, if begin_date='20070101'
        and end_date='20070102', freq='1h', the original dataset will contain data
        from 20070101 00:00 to 20070101 23:00. (24 data points)
        Take `lead_time` into consideration, the final dataset will contain (24 - `lead_time`) data points.
    """
    def __init__(self,
                 data_path,
                 begin_date,
                 end_date,
                 freq='1h',
                 lead_time=1,
                 normalize_data_path=None,
                 cache_len=0,
                 ):
        self.data_path = data_path

        try:
            self.data_keys = list(pd.date_range(start=begin_date, end=end_date, freq=freq, inclusive='left'))
        except:
            # compatible with pandas 1.x
            self.data_keys = list(pd.date_range(start=begin_date, end=end_date, freq=freq, closed='left'))

        # ground truth data keys
        gt_begin_date = pd.to_datetime(begin_date) + pd.DateOffset(hours=lead_time)
        gt_end_date = pd.to_datetime(end_date) + pd.DateOffset(hours=lead_time)
        try:
            self.gt_data_keys = list(pd.date_range(start=gt_begin_date, end=gt_end_date, freq=freq, inclusive='left'))
        except:
            # compatible with pandas 1.x
            self.gt_data_keys = list(pd.date_range(start=gt_begin_date, end=gt_end_date, freq=freq, closed='left'))

        assert len(self.data_keys) == len(self.gt_data_keys)

        final_i = 0
        for i in range(len(self.data_keys) - 1, -1, -1):
            final_i = i
            if self.gt_data_keys[i] < pd.to_datetime(end_date):
                final_i += 1
                break
        assert final_i > 0
        self.data_keys = self.data_keys[:final_i]
        self.gt_data_keys = self.gt_data_keys[:final_i]

        # normalization
        self.normalize = (normalize_data_path is not None)
        self.normalize_data_path = normalize_data_path
        if self.normalize_data_path:
            self.statistic_data = load_data_stats(self.normalize_data_path)

        # cache
        assert cache_len >= 0
        self.cache_len = cache_len
        if self.cache_len > 0:
            self.data_cache = CacheDict(cache_len=2)
        self.cache_hit = 0
        self.cache_miss = 0

    def __len__(self):
        return len(self.data_keys)

    def __getitem__(self, index):
        input_upper, input_surface, target_upper, target_surface = self._load_data(index)
        if self.normalize:
            input_upper, input_surface = compute_norm_data(input_upper, input_surface, self.statistic_data)
            target_upper, target_surface = compute_norm_data(target_upper, target_surface, self.statistic_data)
        return input_upper, input_surface, target_upper, target_surface

    def __repr__(self):
        return self.__class__.__name__

    def _load_data(self, index: int):
        ts = self.data_keys[index]
        gt_ts = self.gt_data_keys[index]

        # foramat to YYYYMMDD
        ts_str = ts.strftime('%Y%m%d')
        gt_ts_str = gt_ts.strftime('%Y%m%d')

        # load data
        input_upper_dataset, input_surface_dataset = self._get_dataset(ts_str)
        target_upper_dataset, target_surface_dataset = self._get_dataset(gt_ts_str)

        input_upper_data = input_upper_dataset.sel(time=ts)
        input_surface_data = input_surface_dataset.sel(time=ts)
        assert input_upper_data['time'] == input_surface_data['time']

        target_upper_data = target_upper_dataset.sel(time=gt_ts)
        target_surface_data = target_surface_dataset.sel(time=gt_ts)
        assert target_upper_data['time'] == target_surface_data['time']

        input_upper, input_surface = era5_nctonumpy(input_upper_data, input_surface_data)
        target_upper, target_surface = era5_nctonumpy(target_upper_data, target_surface_data)
        return input_upper, input_surface, target_upper, target_surface

    def _get_dataset(self, timestamp: str) -> Tuple[xr.Dataset, xr.Dataset]:
        if self.cache_len == 0:
            return self._load_dataset(timestamp)

        # use cache
        if timestamp not in self.data_cache.keys():
            # debug(f'Timestamp {timestamp} cache miss')
            self.cache_miss += 1
            self._load_dataset_to_cache(timestamp)
        else:
            self.cache_hit += 1
            # debug(f'Timestamp {timestamp} cache hit')
        dataset = self.data_cache[timestamp]
        upper_dataset = dataset['upper']
        surface_dataset = dataset['surface']

        return upper_dataset, surface_dataset

    def _load_dataset(self, timestamp: str) -> Tuple[xr.Dataset, xr.Dataset]:
        upper_path = os.path.join(self.data_path, 'upper', f'{timestamp}.nc')
        surface_path = os.path.join(self.data_path, 'surface', f'{timestamp}.nc')

        upper_dataset = xr.open_dataset(upper_path)
        surface_dataset = xr.open_dataset(surface_path)
        return upper_dataset, surface_dataset

    def _load_dataset_to_cache(self, timestamp: str):
        upper_dataset, surface_dataset = self._load_dataset(timestamp)
        dataset  = {
            'upper': upper_dataset,
            'surface': surface_dataset
        }
        self.data_cache[timestamp] = dataset


def era5_nctonumpy(dataset_upper: xr.Dataset, dataset_surface: xr.Dataset) -> Tuple[np.ndarray, np.ndarray]:
    upper_z = dataset_upper['z'].values.astype(np.float32)  # (13, 721, 1440)
    upper_q = dataset_upper['q'].values.astype(np.float32)
    upper_t = dataset_upper['t'].values.astype(np.float32)
    upper_u = dataset_upper['u'].values.astype(np.float32)
    upper_v = dataset_upper['v'].values.astype(np.float32)
    # upper = np.concatenate((upper_z[np.newaxis, ...], upper_q[np.newaxis, ...], upper_t[np.newaxis, ...],
    #                         upper_u[np.newaxis, ...], upper_v[np.newaxis, ...]), axis=0)
    upper = np.stack((upper_z, upper_q, upper_t, upper_u, upper_v), axis=0)
    assert upper.shape == (5, 13, 721, 1440)
    # levels in descending order, require new memery space
    upper = upper[:, ::-1, :, :].copy()

    surface_mslp = dataset_surface['msl'].values.astype(np.float32)  # (721,1440)
    surface_u10 = dataset_surface['u10'].values.astype(np.float32)
    surface_v10 = dataset_surface['v10'].values.astype(np.float32)
    surface_t2m = dataset_surface['t2m'].values.astype(np.float32)
    # surface = np.concatenate((surface_mslp[np.newaxis, ...], surface_u10[np.newaxis, ...],
    #                             surface_v10[np.newaxis, ...], surface_t2m[np.newaxis, ...]), axis=0)
    surface = np.stack((surface_mslp, surface_u10, surface_v10, surface_t2m), axis=0)
    assert surface.shape == (4, 721, 1440)

    return upper, surface

