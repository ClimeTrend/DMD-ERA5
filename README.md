# dmd-era5

[![Actions Status][actions-badge]][actions-link]
[![PyPI version][pypi-version]][pypi-link]
[![PyPI platforms][pypi-platforms]][pypi-link]

Running DMD on ERA5 data

## Installation

```bash
python -m pip install dmd_era5
```

From source:

```bash
git clone https://github.com/ClimeTrend/DMD-ERA5
cd dmd-era5
python -m pip install .
```

## Usage

### Install the package

To install from source:

```bash
git clone git@github.com:ClimeTrend/DMD-ERA5.git
cd DMD-ERA5
python -m venv .venv
source .venv/bin/activate
pip install .
```

### Set up Data Version Control (DVC)

This is an optional step but is recommended.

`dmd-era5` uses [DVC](https://dvc.org/) to manage the data. Most DVC steps are automated, but you will need to set up DVC for the first time. DVC will have been installed as a dependency when you installed this package. DVC works alongside Git, so it might be a good to work in a new branch:

```bash
cd DMD-ERA5  # move to the root directory of the repository if you are not already there
source .venv/bin/activate  # activate the virtual environment if you are not already in it
git checkout -b my-new-branch  # create a new branch
dvc init  # Initialize DVC repository
```

`dvc init` will create a `.dvc` directory and a `.dvcignore` file. You should commit these to your Git repository:

```bash
git status
git commit -m "Initialize DVC"
```

If you plan on collaborating with others, you may want to set up a remote storage location for your data. This is not necessary for personal use, although can be used to back up your data. DVC supports a variety of storage backends, including local storage, cloud storage, etc. For more information, see the [DVC documentation](https://dvc.org/doc/start).

For example, if you are collaborating in a HPC, you might want to use a "local remote" (i.e. a shared directory in your file system). In the `environment.sh` file, change the `DMD_ERA5_DVC_LOCAL_REMOTE` to your desired directory path. Then run:

```bash
source environment.sh
make dvc-local-remote  # will create the directory if it does not exist and set it up as a DVC remote
dvc remote list  # check that the remote was added
```

### Download ERA5 data

Modify the `era5-download` section of the `config.ini` file to specify the desired download parameters. Then run:

```bash
python -m dmd_era5.era5_download.era5_download
```

Note that depending on the size of the download, this may be a memory-intensive process and you might need to run it on a HPC. The downloaded data will be saved in the `data/era5_download` directory.

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
