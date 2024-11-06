# Set up a local remote storage for DVC
# This script will modify the contents of the .dvc/config.local file

# Create a directory for the local remote storage
local_remote="/tmp/dvcstore"  # Change this to your desired directory

if [ -d "$local_remote" ]; then
    echo "Directory $local_remote already exists."
else
    mkdir -p $local_remote
    echo "Directory $local_remote created."
fi

# Add the local remote storage to the DVC local configuration
dvc remote add -d local_remote $local_remote --local
