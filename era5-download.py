# import xarray

# Pseudo code to download ERA5 data from Google Cloud
# with arguments from a config file




# variables = [
#     "u_component_of_wind",
#     "geopotential",
#     "temperature",
#     # ... full list of variables here: https://console.cloud.google.com/storage/browser/gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2;tab=objects?pli=1&prefix=&forceOnObjectsSortingFiltering=false
# ]

# # Loading ERA5 data from Google Cloud arco project
# # see https://cloud.google.com/storage/docs/public-datasets/era5#data_access
# era5_path = (
#     "gs://gcp-public-data-arco-era5/ar/1959-2022-full_37-1h-0p25deg-chunk-1.zarr-v2"
# )

# print("Loading ERA5 data...")
# full_era5_ds = xarray.open_zarr(era5_path, chunks=None)
# print("ERA5 data loaded.")

# full_era5_ds = full_era5_ds[variables]

# start_time = "2020-01-01"
# end_time = "2020-02-27"
# data_inner_steps = 6  # process every 6th hour

# print("Slicing ERA5 data...")
# era5_ds = full_era5_ds.sel(time=slice(start_time, end_time), level=[1000])
# era5_ds = era5_ds.thin(time=data_inner_steps)
# print("ERA5 data sliced.")

# # here, era5_ds will be much smaller, and you can then materialize it/save it/etc.
# print("Saving ERA5 data...")
# era5_ds.to_netcdf(f"{start_time}_{end_time}_era5_slice.nc")
# print("ERA5 data saved.")
