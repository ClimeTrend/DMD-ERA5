import os
from datetime import datetime

import matplotlib.pyplot as plt
import numpy as np
import xarray as xr
from pydmd import BOPDMD
from pydmd.preprocessing import hankel_preprocessing


def run_dmd_analysis(ds, output_dir):
    """
    Run DMD analysis on ERA5 dataset and save results

    Parameters
    ----------
    ds : xarray.Dataset
        Input ERA5 dataset
    output_dir : str
        Directory to save outputs
    """
    # Create output directory if it doesn't exist
    os.makedirs(output_dir, exist_ok=True)

    # 1. Prepare the data
    temp_data = ds["temperature"].isel(level=0)  # Select first level

    # Handle both scalar and array level values
    level_value = (
        temp_data.level.item()
        if temp_data.level.size == 1
        else temp_data.level.values[0]
    )
    print(f"Level chosen: {level_value} hPa")

    X = temp_data.values.reshape(temp_data.shape[0], -1).T  # Reshape to (space, time)

    # Get time vector from xarray and convert to hours since start
    t = (ds.time - ds.time[0]) / np.timedelta64(1, "h")
    t = t.values

    # Get spatial dimensions
    lats = ds.latitude.values
    lons = ds.longitude.values
    weights = np.cos(np.deg2rad(lats))

    # 2. Set up train/test split
    train_frac = 0.8
    T_train = int(len(t) * train_frac)

    # Split data
    X_train = X[:, :T_train]
    t_train = t[:T_train]

    # 3. DMD parameters
    svd_rank = -1  # Make sure it is not trying to optimise to everything
    delay = 2

    # Print the size of the variable
    print(f"size of X: {X_train.shape}")

    # 4. Fit DMD
    optdmd = BOPDMD(
        svd_rank=svd_rank,
        num_trials=0,
        use_proj=True,
        eig_constraints={"imag"},
        varpro_opts_dict={"verbose": True},
    )
    delay_optdmd = hankel_preprocessing(optdmd, d=delay)

    # Adjust time vector for Hankel preprocessing
    t_train_adjusted = t_train[delay - 1 :]

    # Fit DMD with adjusted time vector
    delay_optdmd.fit(X_train, t=t_train_adjusted)

    # 5. Get DMD components
    modes = delay_optdmd.modes
    eigs = delay_optdmd.eigs
    amplitudes = delay_optdmd.amplitudes

    # 6. Create time vector and compute DMD solution
    n_points = X.shape[1]
    t_eval = np.arange(n_points) * (t[1] - t[0])  # Use actual time step

    vander = np.vander(eigs, n_points, increasing=True)
    X_dmd = (modes @ np.diag(amplitudes) @ vander).T

    # 7. Reshape and compute spatial means
    n_spatial = X.shape[0]
    n_time = X.shape[1]
    n_lat = len(lats)
    n_lon = len(lons)

    X_dmd = X_dmd[:n_time, :n_spatial].T

    # Compute weighted spatial means
    X_true_mean = np.average(
        np.average(
            X.reshape(n_spatial, n_time).reshape(-1, n_lat, n_lon),
            weights=weights,
            axis=1,
        ),
        axis=1,
    )
    X_dmd_mean = np.average(
        np.average(X_dmd.reshape(-1, n_lat, n_lon), weights=weights, axis=1), axis=1
    )

    # 8. Create and save plot
    plt.figure(figsize=(10, 6))
    plt.plot(t_eval, X_dmd_mean, color="grey", label="DMD reconstruction/prediction")
    plt.plot(t_eval, X_true_mean, color="r", label="True values")
    plt.axvline(t_eval[T_train], linestyle="--", color="k", label="Present")
    plt.ylabel("Spatial mean temperature (K)")
    plt.xlabel("Hours")
    plt.legend()
    plt.title("DMD Reconstruction and Prediction")

    # Save plot
    timestamp = datetime.now().strftime("%Y%m%d_%H%M%S")
    plt.savefig(os.path.join(output_dir, f"dmd_prediction_{timestamp}.png"))
    plt.close()

    # 9. Save DMD results as numpy arrays
    np.save(os.path.join(output_dir, f"dmd_modes_{timestamp}.npy"), modes)
    np.save(os.path.join(output_dir, f"dmd_eigs_{timestamp}.npy"), eigs)
    np.save(os.path.join(output_dir, f"dmd_amplitudes_{timestamp}.npy"), amplitudes)
    np.save(os.path.join(output_dir, f"dmd_prediction_{timestamp}.npy"), X_dmd)

    # 10. Save metadata
    metadata = {
        "train_frac": train_frac,
        "svd_rank": svd_rank,
        "delay": delay,
        "n_modes": modes.shape[1],
        "spatial_shape": (n_lat, n_lon),
        "temporal_points": n_time,
        "train_points": T_train,
    }
    np.save(os.path.join(output_dir, f"dmd_metadata_{timestamp}.npy"), metadata)

    return timestamp


if __name__ == "__main__":
    # Get the current directory
    current_dir = os.path.dirname(os.path.abspath(__file__))

    # Path to your ERA5 data (adjust these paths based on your HPC structure)
    data_path = os.path.join(
        current_dir, "data", "era5_download", "2019-01-01T00_2019-01-05T00_1h.nc"
    )
    output_dir = os.path.join(current_dir, "data", "dmd_results")

    # Load data
    ds = xr.open_dataset(data_path)

    # Run analysis
    timestamp = run_dmd_analysis(ds, output_dir)

    print(f"Analysis complete. Results saved with timestamp: {timestamp}")
