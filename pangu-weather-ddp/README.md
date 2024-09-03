# pangu-weather-ddp

still under working...

Ref:
- https://github.com/198808xc/Pangu-Weather
- https://github.com/zhaoshan2/pangu-pytorch


## Data

upper-air:
https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-pressure-levels?tab=form

surface:
https://cds.climate.copernicus.eu/cdsapp#!/dataset/reanalysis-era5-single-levels?tab=form

use cdsapi to download data:
https://cds.climate.copernicus.eu/api-how-to

> Note: (1) The variables to choose can be found in the original paper.
(2) Download the NetCDF format data.

one day data size (2007.01.01, all 24 hours):
- upper-air: 3.02GB
- surface: 190.1MB

## Code

python packages required:
```
pytorch
xarray
timm
pandas
cdsapi  # For download ERA5 dataset
psutil  # For memory monitoring
```
