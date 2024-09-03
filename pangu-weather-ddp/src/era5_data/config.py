import sys
from .ordered_easydict import OrderedEasyDict as edict
import numpy as np
import os
import torch

__C = edict()
cfg = __C
__C.GLOBAL = edict()

# TODO: Change the path to the root of the project
__C.GLOBAL.PATH = "/Users/wu/Documents/Code/python/pangu_weather/pangu-weather-ddp"
assert os.path.exists(__C.GLOBAL.PATH)

__C.PG_CONST_MASK_PATH = os.path.join(__C.GLOBAL.PATH, 'constant_masks')

# TODO: Put the original data in the data folder
__C.PG_INPUT_PATH = os.path.join(__C.GLOBAL.PATH, 'data')
assert os.path.exists(__C.PG_INPUT_PATH)

__C.PG_OUT_PATH = os.path.join(__C.GLOBAL.PATH,'result')
assert __C.PG_OUT_PATH is not None

__C.ERA5_UPPER_LEVELS = ['1000','925','850', '700','600','500','400', '300','250', '200','150','100', '50']
__C.ERA5_SURFACE_VARIABLES = ['msl','u10','v10','t2m']
__C.ERA5_UPPER_VARIABLES = ['z','q','t','u','v']


__C.PG = edict()

__C.PG.TRAIN = edict()

__C.PG.HORIZON = 24

__C.PG.TRAIN.EPOCHS = 100
__C.PG.TRAIN.LR = 5e-4 # 5e-6 #5e-4
__C.PG.TRAIN.WEIGHT_DECAY = 3e-6
__C.PG.TRAIN.START_TIME = '20150101'
__C.PG.TRAIN.END_TIME = '20171231' #'20171231'
__C.PG.TRAIN.FREQUENCY = '12H'
__C.PG.TRAIN.BATCH_SIZE = 2
__C.PG.TRAIN.UPPER_WEIGHTS = [3.00, 0.60, 1.50, 0.77, 0.54]
__C.PG.TRAIN.SURFACE_WEIGHTS = [1.50, 0.77, 0.66, 3.00]
__C.PG.TRAIN.SAVE_INTERVAL = 1
__C.PG.VAL = edict()


__C.PG.VAL.START_TIME = '20190101'
__C.PG.VAL.END_TIME = '20191231'
__C.PG.VAL.FREQUENCY = '12H'
__C.PG.VAL.BATCH_SIZE = 1
__C.PG.VAL.INTERVAL = 1


__C.PG.TEST = edict()

__C.PG.TEST.START_TIME = '20180103'
__C.PG.TEST.END_TIME = '20180117'
__C.PG.TEST.FREQUENCY = '12H'
__C.PG.TEST.BATCH_SIZE = 1

__C.PG.BENCHMARK = edict()

__C.PG.BENCHMARK.PRETRAIN_24 = os.path.join(__C.PG_INPUT_PATH , 'pretrained_model/pangu_weather_24.onnx')
__C.PG.BENCHMARK.PRETRAIN_6 = os.path.join(__C.PG_INPUT_PATH , 'pretrained_model/pangu_weather_6.onnx')
__C.PG.BENCHMARK.PRETRAIN_3 = os.path.join(__C.PG_INPUT_PATH , 'pretrained_model/pangu_weather_3.onnx')
__C.PG.BENCHMARK.PRETRAIN_1 = os.path.join(__C.PG_INPUT_PATH , 'pretrained_model/pangu_weather_1.onnx')

__C.PG.BENCHMARK.PRETRAIN_24_fp16 = os.path.join(__C.PG_INPUT_PATH , 'pretrained_model_fp16/pangu_weather_24_fp16.onnx')

__C.PG.BENCHMARK.PRETRAIN_24_torch = os.path.join(__C.PG_INPUT_PATH , 'pretrained_model/pangu_weather_24_torch.pth')


__C.MODEL = edict()


