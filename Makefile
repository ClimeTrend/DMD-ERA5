# Makefile

.PHONY: pytest pytest-docker-build pytest-docker-run dvc-local-remote repo-docker-build

# Build Docker image from Dockerfile to run the repository
repo-docker-build:
	docker image rm -f $(DMD_ERA5_IMAGE_NAME_REPO)
	docker build -t $(DMD_ERA5_IMAGE_NAME_REPO) -f Dockerfile.repo .

# Run tests except those marked as "docker"
pytest:
	pytest -v

# Build Docker image from Dockerfile to run tests marked as "docker"
pytest-docker-build:
	docker image rm -f $(DMD_ERA5_IMAGE_NAME_PYTEST)
	docker build -t $(DMD_ERA5_IMAGE_NAME_PYTEST) -f Dockerfile.pytest .

# Run tests marked as "docker" in Docker container
pytest-docker-run:
	docker run --rm $(DMD_ERA5_IMAGE_NAME_PYTEST)

# Initialize DVC
dvc-init:
	dvc init

# Create DVC local remote
dvc-local-remote:
	@if [ -d $(DMD_ERA5_DVC_LOCAL_REMOTE) ]; then \
    	echo "Directory $(DMD_ERA5_DVC_LOCAL_REMOTE) already exists."; \
	else \
    	mkdir -p $(DMD_ERA5_DVC_LOCAL_REMOTE); \
    	echo "Directory $(DMD_ERA5_DVC_LOCAL_REMOTE) created."; \
	fi
	@if dvc remote list | grep -q "^local_remote"; then \
		echo "DVC remote 'local_remote' already exists:"; \
		dvc remote list; \
	else \
		dvc remote add -d local_remote $(DMD_ERA5_DVC_LOCAL_REMOTE) --local; \
		echo "DVC remote 'local_remote' added."; \
	fi

# Set up DVC autostage
dvc-autostage:
	dvc config core.autostage true
	git add .dvc/config

# Full DVC setup
dvc-setup: dvc-init dvc-local-remote dvc-autostage
