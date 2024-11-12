# Makefile
IMAGE_NAME_REPO = dmd_era5  # Docker image name for the repository
IMAGE_NAME_PYTEST = dmd_era5_pytest  # Docker image name to run pytest
LOCAL_REMOTE = "/tmp/dvcstore"  # Local remote directory for DVC, change to your desired path

.PHONY: pytest pytest-docker-build pytest-docker-run dvc-local-remote repo-docker-build

# Build Docker image from Dockerfile to run the repository
repo-docker-build:
	docker build -t $(IMAGE_NAME_REPO) -f Dockerfile.repo .

# Run tests except those marked as "docker"
pytest:
	pytest -v

# Build Docker image from Dockerfile to run tests marked as "docker"
pytest-docker-build:
	docker build -t $(IMAGE_NAME_PYTEST) -f Dockerfile.pytest .

# Run tests marked as "docker" in Docker container
pytest-docker-run:
	docker run $(IMAGE_NAME_PYTEST)

# Create DVC local remote
dvc-local-remote:
	@if [ -d $(LOCAL_REMOTE) ]; then \
    	echo "Directory $(LOCAL_REMOTE) already exists."; \
	else \
    	mkdir -p $(LOCAL_REMOTE); \
    	echo "Directory $(LOCAL_REMOTE) created."; \
	fi
	@if dvc remote list | grep -q "^local_remote"; then \
		echo "DVC remote 'local_remote' already exists:"; \
		dvc remote list; \
	else \
		dvc remote add -d local_remote $(LOCAL_REMOTE) --local; \
		echo "DVC remote 'local_remote' added."; \
	fi
