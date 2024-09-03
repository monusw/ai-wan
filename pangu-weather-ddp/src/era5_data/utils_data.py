import xarray as xr
from datetime import datetime, timedelta
import pandas as pd
import numpy as np
from era5_data.config import cfg

from typing import Tuple, List
import torch
import random
from torch.utils import data
import os

class DataPrefetcher():
    def __init__(self, loader):
        self.loader = loader
        self.dataiter = iter(loader)
        self.length = len(self.loader)
        self.stream = torch.cuda.Stream()
        # self.mean = torch.tensor([0.485 * 255, 0.456 * 255, 0.406 * 255]).cuda().view(1,3,1,1)
        # self.std = torch.tensor([0.229 * 255, 0.224 * 255, 0.225 * 255]).cuda().view(1,3,1,1)
        # With Amp, it isn't necessary to manually convert data to half.
        # if args.fp16:
        #     self.mean = self.mean.half()
        #     self.std = self.std.half()
        self.__preload__()

    def __preload__(self):
        try:
            self.input, self.input_surface, self.target, self.target_surface, self.periods = next(self.dataiter)
        except StopIteration:
            self.dataiter = iter(self.loader)
            self.input, self.input_surface, self.target, self.target_surface, self.periods = next(self.dataiter)

        with torch.cuda.stream(self.stream):
            self.target = self.target.cuda(non_blocking=True)
            self.target_surface = self.target_surface.cuda(non_blocking=True)
            self.input = self.input.cuda(non_blocking=True)
            self.input_surface = self.input_surface.cuda(non_blocking=True)
            self.periods = self.periods.cuda(non_blocking=True)

    def next(self):
        torch.cuda.current_stream().wait_stream(self.stream)
        self.__preload__()
        return self.input, self.input_surface, self.target, self.target_surface, self.periods

    def __len__(self):
        """Return the number of images."""
        return self.length


def weatherStatistics_output(filepath="/home/code/data_storage_home/data/pangu/aux_data", device="cpu"):
    """
    :return:1, 5, 13, 1, 1
    """
    surface_mean = np.load(os.path.join(filepath, "surface_mean.npy")).astype(np.float32)
    surface_std = np.load(os.path.join(filepath, "surface_std.npy")).astype(np.float32)
    surface_mean = torch.from_numpy(surface_mean)
    surface_std = torch.from_numpy(surface_std)
    surface_mean = surface_mean.view(1, 4, 1, 1)
    surface_std = surface_std.view(1, 4, 1, 1)

    upper_mean = np.load(os.path.join(filepath, "upper_mean.npy")).astype(np.float32)  # (13,1,1,5)
    upper_mean = upper_mean[::-1, :, :, :].copy()
    upper_mean = np.transpose(upper_mean, (1, 3, 0, 2))  # (1,5,13, 1)
    upper_mean = torch.from_numpy(upper_mean)

    upper_std = np.load(os.path.join(filepath, "upper_std.npy")).astype(np.float32)
    upper_std = upper_std[::-1, :, :, :].copy()
    upper_std = np.transpose(upper_std, (1, 3, 0, 2))
    upper_std = torch.from_numpy(upper_std)

    return surface_mean.to(device), surface_std.to(device), upper_mean[..., None].to(device), upper_std[..., None].to(
        device)


def weatherStatistics_input(filepath="/home/code/data_storage_home/data/pangu/aux_data", device="cpu"):
    """
    :return:13, 1, 1, 5
    """
    surface_mean = np.load(os.path.join(filepath, "surface_mean.npy")).astype(np.float32)
    surface_std = np.load(os.path.join(filepath, "surface_std.npy")).astype(np.float32)
    surface_mean = torch.from_numpy(surface_mean)
    surface_std = torch.from_numpy(surface_std)

    upper_mean = np.load(os.path.join(filepath, "upper_mean.npy")).astype(np.float32)
    upper_std = np.load(os.path.join(filepath, "upper_std.npy")).astype(np.float32)
    upper_mean = torch.from_numpy(upper_mean)
    upper_std = torch.from_numpy(upper_std)

    return surface_mean.to(device), surface_std.to(device), upper_mean.to(device), upper_std.to(device)


def LoadConstantMask(filepath: str, device: torch.device, batch_size: int=1):
    assert batch_size > 0
    land_mask = np.load(os.path.join(filepath, "land_mask.npy")).astype(np.float32)
    soil_type = np.load(os.path.join(filepath, "soil_type.npy")).astype(np.float32)
    topography = np.load(os.path.join(filepath, "topography.npy")).astype(np.float32)
    land_mask = torch.from_numpy(land_mask)  # ([721, 1440])
    soil_type = torch.from_numpy(soil_type)  # ([721, 1440])
    topography = torch.from_numpy(topography)  # ([721, 1440])

    land_mask = land_mask[None, None, ...].repeat(batch_size, 1, 1, 1)    # [batch_size, 1, 721, 1440]
    soil_type = soil_type[None, None, ...].repeat(batch_size, 1, 1, 1)    # [batch_size, 1, 721, 1440]
    topography = topography[None, None, ...].repeat(batch_size, 1, 1, 1)  # [batch_size, 1, 721, 1440]

    return land_mask.to(device), soil_type.to(device), topography.to(device)


def LoadConstantMask3(filepath="/home/code/data_storage_home/data/pangu/aux_data", device="cpu"):
    mask = np.load(os.path.join(filepath, "constantMaks3.npy")).astype(np.float32)
    mask = torch.from_numpy(mask)
    return mask.to(device)


def compute_statistics(train_loader):
    # prepare for the statistics
    weather_surface_mean, weather_surface_std = torch.zeros(1, 4, 1, 1), torch.zeros(1, 4, 1, 1)
    weather_mean, weather_std = torch.zeros(1, 5, 13, 1, 1), torch.zeros(1, 5, 13, 1, 1)
    total = len(train_loader)
    for id, train_data in enumerate(train_loader, 0):
        print(f"\r{id+1}/{total} data computed", end="")
        input, input_surface, *_ = train_data
        weather_surface_mean += torch.mean(input_surface, dim=(-1, -2), keepdim=True)
        weather_surface_std += torch.std(input_surface, dim=(-1, -2), keepdim=True)
        weather_mean += torch.mean(input, dim=(-1, -2), keepdim=True)
        weather_std += torch.std(input, dim=(-1, -2), keepdim=True)  # (1,5,13,)
    weather_surface_mean, weather_surface_std, weather_mean, weather_std = \
        weather_surface_mean / len(train_loader), weather_surface_std / len(train_loader), weather_mean / len(
            train_loader), weather_std / len(train_loader)
    print()

    # surface: (4, 1, 1)
    # upper: (5, 13, 1, 1)
    return (torch.squeeze(weather_surface_mean, 0),
            torch.squeeze(weather_surface_std, 0),
            torch.squeeze(weather_mean, 0),
            torch.squeeze(weather_std, 0))

def compute_and_save_data_stats(data_loader, outpath):
    surface_mean, surface_std, upper_mean, upper_std = compute_statistics(data_loader)
    data = {
        'upper_mean': upper_mean,
        'upper_std': upper_std,
        'surface_mean': surface_mean,
        'surface_std': surface_std,
    }
    np.savez(outpath, **data)

def load_data_stats(filepath):
    data = np.load(filepath)
    return data

def compute_norm_data(upper: np.ndarray, surface: np.ndarray, statistics):
    upper_mean = statistics['upper_mean']
    upper_std = statistics['upper_std']
    surface_mean = statistics['surface_mean']
    surface_std = statistics['surface_std']

    upper = (upper - upper_mean) / upper_std
    surface = (surface - surface_mean) / surface_std
    return upper, surface

def loadConstMask_h(filepath="/home/code/data_storage_home/data/pangu/aux_data", device="cpu"):
    mask_h = np.load(os.path.join(filepath, "Constant_17_output_0.npy")).astype(np.float32)
    mask_h = torch.from_numpy(mask_h)
    return mask_h.to(device)


def load_variable_weights(device="cpu"):
    upper_weights = torch.FloatTensor(cfg.PG.TRAIN.UPPER_WEIGHTS).unsqueeze(0).unsqueeze(2).unsqueeze(3).unsqueeze(4)
    surface_weights = torch.FloatTensor(cfg.PG.TRAIN.SURFACE_WEIGHTS).unsqueeze(0).unsqueeze(2).unsqueeze(3)
    return upper_weights.to(device), surface_weights.to(device)


def loadAllConstants(device):
    constants = dict()
    constants['weather_statistics'] = weatherStatistics_input(
        device=device)  # height has inversed shape, order is reversed in model
    constants['weather_statistics_last'] = weatherStatistics_output(device=device)
    # constants['constant_maps'] = LoadConstantMask(device=device)
    constants['constant_maps'] = LoadConstantMask3(device=device) #not able to be equal
    constants['variable_weights'] = load_variable_weights(device=device)
    constants['const_h'] = loadConstMask_h(device=device)

    return constants

def normData(upper, surface, statistics):
    surface_mean, surface_std, upper_mean, upper_std = (
        statistics[0], statistics[1], statistics[2], statistics[3])

    upper = (upper - upper_mean) / upper_std
    surface = (surface - surface_mean) / surface_std
    return upper, surface


def normBackData(upper, surface, statistics):
    surface_mean, surface_std, upper_mean, upper_std = (
        statistics[0], statistics[1], statistics[2], statistics[3])
    upper = upper * upper_std + upper_mean
    surface = surface * surface_std + surface_mean

    return upper, surface

if __name__ == "__main__":
    # dataset_path ='/home/code/data_storage_home/data/pangu'
    # means, std = LoadStatic(os.path.join(dataset_path, 'aux_data'))
    # print(means.shape) #(1, 21, 1, 1)
    a, b, c, d = weatherStatistics_input()
    print(a.shape)

