# dmd-era5

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

Running DMD on ERA5 data

## Installation

From source:

```bash
git clone https://github.com/ClimeTrend/DMD-ERA5
cd DMD-ERA5
python -m venv .venv
source .venv/bin/activate
python -m pip install .
```

## Usage

### Set up Data Version Control (DVC)

This is an optional step, but is highly recommended. Note that you only need to run through this process once.

`dmd-era5` uses [DVC](https://dvc.org/) to manage data versioning. DVC allows to keep multiple versions of a dataset with the same file name, and to set up a remote storage location to share datasets between project collaborators. Most DVC steps are automated, but you will need to set up DVC for the first time. DVC will have been installed as a dependency when you installed this package. DVC works alongside Git to track changes to data files, so it is a good idea to work in a new branch off the main branch when using `dmd-era5` with DVC:

```bash
git checkout -b my-working-branch
```

If you plan on collaborating with others, you may want to set up a remote storage location for your data. This is not necessary for personal use, although can be used to back up your data. DVC supports a variety of storage backends, including local storage, cloud storage, etc. For more information, see the [DVC documentation](https://dvc.org/doc/start). For example, if you are collaborating in a HPC, you might want to use a "local remote" (i.e. a shared directory in your file system). To set up a local remote follow these steps:

1. In the `environment.sh` file, change the `DMD_ERA5_DVC_LOCAL_REMOTE` environment variable to your desired directory path
2. Export the environment variables by running:

    ```bash
    source environment.sh
    ```

3. To perform the complete DVC setup, run:

    ```bash
    make dvc-setup
    ```

    Which will initialize DVC, set up the local remote, and set DVC auto-stage to `true`. If you don't want to set up the local remote, instead you can run:

    ```bash
    make dvc-init
    make dvc-autostage
    ```

4. The previous command will have created a `.dvc` directory and a `.dvcignore` file. You will also see a few new files in the Git staging area. Commit these changes to Git with a message like "Set up DVC". Now you are ready to start using `dm-era5` with DVC.

### Download ERA5 data

Modify the `era5-download` section of the `config.ini` file to specify the desired download parameters. Then run:

```bash
python -m dmd_era5.era5_download.era5_download
```

Note that depending on the size of the download, this may be a memory-intensive process and you might need to run it on a HPC. The downloaded data will be saved in the `data/era5_download` directory as a NetCDF file, using the time range and time delta specified in `config.ini` as the file name (e.g. `2019-01-01T00_2019-01-02T00_1h.nc` for a time range from 2019-01-01 00:00 to 2019-01-02 00:00 with a time delta of 1 hour).

If you followed the DVC setup instructions above, the downloaded data will be automatically tracked by DVC, and you will see three new files appearing in your Git staging area. For example, if you downloaded the file `2019-01-01T00_2019-01-02T00_1h.nc`, you will see the following files in the Git staging area:

- `data/era5_download/.gitignore`, automatically created by DVC to tell Git to ignore the downloaded data file.
- `data/era5_download/2019-01-01T00_2019-01-02T00_1h.nc.dvc`, automatically created by DVC to track the downloaded data version with a unique hash.
- `data/era5_download/2019-01-01T00_2019-01-02T00_1h.nc.yaml`, which is a human-readable log of the different versions of the downloaded data, showing the metadata of each version. You can open this file to see the different versions of `2019-01-01T00_2019-01-02T00_1h.nc` that you have downloaded.

You should now commit these files to Git with a message like "First download of 2019-01-01T00_2019-01-02T00_1h.nc". If you have set up a remote storage location for your data, you can push the data to the remote storage location by running:

```bash
dvc push
```

Once you have pushed to the remote, it is safe to delete the downloaded NetCDF file from `data/era5_download` if you wish, and you can even delete the DVC cache (which stores the data versions locally) by running `rm -rf .dvc/cache`. Note that deleting the data stored locally is an optional step that might be useful if you are running out of disk space.

Next time you want to download the same data or a subset of it (i.e. a subset of variables or pressure levels) over the same time range and time delta and from the same source, you can simply run `python -m dmd_era5.era5_download.era5_download` again. DVC will automatically detect that you have already downloaded the data and will not download it again; it will simply retrieve the right version of the data from the DVC repository.

### Perform SVD on ERA5

This package makes use of the optimized DMD algorithm proposed by Askham and Kutz (2018). An optional step of the algorithm that is suitable for large datasets is to perform Singular Value Decomposition (SVD) on the data prior to running DMD. `dmd-era5` allows you to perform SVD on ERA5 data by running:

```bash
python -m dmd_era5.era5_svd.era5_svd
```

Modify the `era5-svd` section of the `config.ini` file to specify the desired SVD parameters. The SVD will be performed on the downloaded ERA5 data, which is assumed to be stored in the `data/era5_download` directory. The SVD results will be saved in the `data/era5_svd` directory as a NetCDF file, using the time range and time delta specified in `config.ini` as the file name (e.g. `2019-01-01T00_2019-01-02T00_1h.nc` for a time range from 2019-01-01 00:00 to 2019-01-02 00:00 with a time delta of 1 hour).

If you followed the DVC setup instructions above, the SVD results will be automatically tracked by DVC, and you will see three new files appearing in your Git staging area, in a similar way to the downloaded data. You should commit these files to Git with a message like "First SVD of 2019-01-01T00_2019-01-02T00_1h.nc", and optionally push them to the remote storage location.

`era5_svd` allows you to perform standard SVD from NumPy or randomized SVD from scikit-learn. The randomized SVD is faster and more memory-efficient than the standard SVD, and it's recommended for large datasets.

#### Contents of the SVD NetCDF file

An example of the contents of the SVD NetCDF file is as follows:

![SVD NetCDF file](media/svd_netcdf_contents.png)

The data variables `U`, `s`, and `V` correspond to the left singular vectors, singular values, and right singular vectors, respectively. Their dimensions are `(space, components)`, `(components,)`, and `(components, time)`, respectively. The components are organized in decreasing singular value order. The spatial dimension results from the flattening of the `latitude`, `longitude`, and `level` dimensions of the original ERA5 data. `space` is just a monotonically increasing index that represents the flattened spatial dimensions. `latitude`, `longitude`, and `level` are kept as coordinates in the NetCDF file, so that the spatial dimensions can be reconstructed from the `space` dimension. For example, to select all longitude values at a specific latitude and level, you can use the `space` dimension to select the corresponding indices:

```python
U = data.U  # get the left singular vectors
U = U.sel(space=U.level == 100)  # select all lat and lon values at level 100
U = U.sel(space=U.latitude == 0)  # select all lon values at latitude 0
```

The coordinate `original_variable` indicates the indices along the `space` dimension that correspond to each of the variables on which the SVD was performed. Variables are treated as additional data points along the `space` dimension, and the `original_variable` coordinate allows to identify which variable each data point corresponds to. If the SVD was performed on a single variable, the `original_variable` coordinate will have a single unique value. If the SVD was performed on multiple variables, the `original_variable` coordinate will have multiple unique values, one for each variable. Similarly, the `delay` coordinate indicates the indices along the spatial dimension that correspond to each of the time delay embeddings applied to the data prior to the SVD. For example, if a time delay embedding of 2 was applied to the data, the `delay` coordinate will have two unique values, 0 and 1, where 0 corresponds to the snapshot at the current `time` value and 1 corresponds to the snapshot at the previous `time` value. The following code snippet shows an example:

```python
U = data.U  # get the left singular vectors
U = U.sel(space=U.original_variable == 'temperature')  # select all lat, lon, and level values for the temperature variable
U = U.sel(space=U.delay == 0)  # select all data points that correspond to the "current" snapshot
```

The attributes of the NetCDF file contain metadata about the SVD results, summarizing the parameters specified in `config.ini`. Note that if the `mean_center` and `scale` parameters are set to `True`, the SVD will be performed on the mean-centered and scaled data matrix, and two additional variables corresponding to the mean (`X_mean`) and standard deviation (`X_std`) of the data will be included in the NetCDF file. Similarly, if the parameter `save_data_matrix` is set to `True`, the preprocessed data matrix on which the SVD was performed will be included in the NetCDF file as the variable `X`. Note that `X` might be very large, in which case it will increase the size of the NetCDF file significantly.

## Contributing

See [CONTRIBUTING.md](CONTRIBUTING.md) for instructions on how to contribute.

## License

Distributed under the terms of the [MIT license](LICENSE).

<!-- prettier-ignore-start -->
[actions-badge]:            https://github.com/ClimeTrend/DMD-ERA5/workflows/CI/badge.svg
[actions-link]:             https://github.com/ClimeTrend/DMD-ERA5/actions
[pypi-link]:                https://pypi.org/project/dmd-era5/
[pypi-platforms]:           https://img.shields.io/pypi/pyversions/dmd-era5
[pypi-version]:             https://img.shields.io/pypi/v/dmd-era5
<!-- prettier-ignore-end -->

## References

1. T. Askham and J. N. Kutz, "Variable Projection Methods for an Optimized Dynamic Mode Decomposition", 2018.
