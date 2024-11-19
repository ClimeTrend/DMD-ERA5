See the [Scientific Python Developer Guide][spc-dev-intro] for a detailed
description of best practices for developing scientific packages.

[spc-dev-intro]: https://learn.scientific-python.org/development/

# Setting up a development environment manually

You can set up a development environment by running:

```zsh
python3 -m venv venv          # create a virtualenv called venv
source ./venv/bin/activate   # now `python` points to the virtualenv python
pip install -v -e ".[dev]"    # -v for verbose, -e for editable, [dev] for dev dependencies
```

# Post setup

You should prepare pre-commit, which will help you by checking that commits pass
required checks.
Pre-commit will have been installed as a dev dependency, so you can run:

```bash
pre-commit install
```

This will install a pre-commit hook into the Git repo, which will run the checks with each Git commit.

You can also/alternatively run `pre-commit run` (changes only) or
`pre-commit run --all-files` to check even without installing the hook.

# Testing

Use pytest to run the unit checks:

```bash
pytest  # will run all tests but the ones marked as "docker"
```

By default, this will skip the tests marked as "docker", which are Data Version Control (DVC) tests designed to run in a Docker container because they modify the environment. To run these tests, start Docker Desktop and then run:

```bash
source environment.sh  # set the environment variables
make repo-docker-build  # build the Docker image for the repo
make pytest-docker-build  # build the Docker image for the tests
make pytest-docker-run  # run the tests marked as "docker" in the Docker container
make pytest  # this is equivalent to just running `pytest`, i.e. it will run all tests but the ones marked as "docker"
```

If you don't have Docker Desktop installed, you can download it from [here](https://www.docker.com/products/docker-desktop).

# Coverage

Use pytest-cov to generate coverage reports:

```bash
pytest --cov=dmd_era5
```
