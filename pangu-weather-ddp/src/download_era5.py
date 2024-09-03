import cdsapi   # put url and key in ~/.cdsapirc
import os
import pandas as pd

from era5_data.config import cfg

def download_surface_data(client: cdsapi.Client, timeinfo):
    folder = 'surface'
    outdir = os.path.join(cfg.PG_INPUT_PATH, folder)
    os.makedirs(outdir, exist_ok=True)

    year = timeinfo['year']
    month = timeinfo['month']
    day = timeinfo['day']
    time = timeinfo['time']

    filename = f"{year}{month}{day}.nc"
    outpath = os.path.join(outdir, filename)

    if os.path.exists(outpath):
        print(f"File {outpath} already exists.")
        return

    client.retrieve(
        'reanalysis-era5-single-levels',
        {
            'product_type': 'reanalysis',
            'variable': [
                '10m_u_component_of_wind', '10m_v_component_of_wind', '2m_temperature',
                'mean_sea_level_pressure',
            ],
            'year': year,
            'month': month,
            'day': day,
            'time': time,
            'format': 'netcdf',
        },
        outpath,
    )

def download_upperair_data(client: cdsapi.Client, timeinfo):
    folder = 'upper'
    outdir = os.path.join(cfg.PG_INPUT_PATH, folder)
    os.makedirs(outdir, exist_ok=True)

    year = timeinfo['year']
    month = timeinfo['month']
    day = timeinfo['day']
    time = timeinfo['time']

    filename = f"{year}{month}{day}.nc"
    outpath = os.path.join(outdir, filename)

    if os.path.exists(outpath):
        print(f"File {outpath} already exists.")
        return

    client.retrieve(
        'reanalysis-era5-pressure-levels',
        {
            'product_type': 'reanalysis',
            'format': 'netcdf',
            'variable': [
                'geopotential', 'specific_humidity', 'temperature',
                'u_component_of_wind', 'v_component_of_wind',
            ],
            'pressure_level': [
                '50', '100', '150',
                '200', '250', '300',
                '400', '500', '600',
                '700', '850', '925',
                '1000',
            ],
            'year': year,
            'month': month,
            'day': day,
            'time': time,
        },
        outpath,
    )


def main():
    client = cdsapi.Client()

    time = [f"{i:02}:00" for i in range(24)]

    beg_date = '20070102'
    end_date = '20070103'
    date_range = pd.date_range(start=beg_date, end=end_date, freq='1D', inclusive='left')
    for date in date_range:
        year = date.strftime('%Y')
        month = date.strftime('%m')
        day = date.strftime('%d')
        print(year, month, day)
        timeinfo = {
            'year': year,
            'month': month,
            'day': day,
            'time': time,
        }

        download_upperair_data(client, timeinfo)
        download_surface_data(client, timeinfo)


if __name__ == "__main__":
    main()

