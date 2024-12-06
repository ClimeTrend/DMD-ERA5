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

    # Add diagnostics
    print(f"Temperature range: {temp_data.min().values} to {temp_data.max().values} K")
    print(f"Temperature standard deviation: {temp_data.std().values} K")

    # Reshape to (space, time)
    X = temp_data.values.reshape(temp_data.shape[0], -1).T
    print(f"X matrix range: {X.min()} to {X.max()}")
    print(f"X matrix standard deviation: {X.std()}")

    # Get time vector from xarray and convert to hours since start
    t = (ds.time - ds.time[0]) / np.timedelta64(1, "h")
    t = t.values

    print(f"Number of total hours: {len(t)}")
    print(f"Number of days: {len(t)/24}")
    print(f"Number of total spatial points: {X.shape[0]}")

    # 2. Set up train/test split FIRST
    train_frac = 0.8
    T_train = int(len(t) * train_frac)

    # Split data before any preprocessing
    X_train = X[:, :T_train]
    X_test = X[:, T_train:]
    X_test = X[:, T_train:]
    t_train = t[:T_train]
    t_test = t[T_train:]

    # 3. DMD parameters
    svd_rank = 10  # Increased from 6
    delay = 2  # Increased from 2

    # Print the size of the variable
    print(f"size of X: {X_train.shape}")

    # Normalize training data only
    X_train_mean = np.mean(X_train, axis=1, keepdims=True)
    X_train_std = np.std(X_train, axis=1, keepdims=True)
    X_train_normalized = (X_train - X_train_mean) / X_train_std

    # 4. DMD parameters
    svd_rank = 6  # Keep this the same for now
    delay = 2

    # 5. Fit DMD
    optdmd = BOPDMD(
        svd_rank=svd_rank,
        num_trials=0,
        use_proj=True,
        eig_constraints={"imag"},
        varpro_opts_dict={
            "verbose": True,
            "maxiter": 100,  # Increase the number of iterations
            "tol": 1e-6,
        },
    )
    delay_optdmd = hankel_preprocessing(optdmd, d=delay)

    # Adjust time vector for Hankel preprocessing
    t_train_adjusted = t_train[delay - 1 :]

    # Fit DMD with adjusted time vector
    delay_optdmd.fit(X_train_normalized, t=t_train_adjusted)

    # 6. Get DMD components
    modes = delay_optdmd.modes
    eigs = delay_optdmd.eigs
    amplitudes = delay_optdmd.amplitudes

    # Get spatial dimensions
    n_spatial = X.shape[0]

    print("\nShape diagnostics:")
    print(f"Original X shape: {X.shape}")
    print(f"Modes shape: {modes.shape}")
    print(f"X_mean shape: {X_train_mean.shape}")
    print(f"X_std shape: {X_train_std.shape}")

    # Separate reconstruction for training and prediction
    # Training period
    vander_train = np.vander(eigs, T_train, increasing=True)
    X_dmd_train_normalized = (modes @ np.diag(amplitudes) @ vander_train).T
    X_dmd_train = (X_dmd_train_normalized * X_train_std.T) + X_train_mean.T

    # Test period (true prediction)
    vander_test = np.vander(eigs, len(t_test), increasing=True)
    X_dmd_test_normalized = (modes @ np.diag(amplitudes) @ vander_test).T
    X_dmd_test = (X_dmd_test_normalized * X_train_std.T) + X_train_mean.T

    # Combine results
    X_dmd = np.concatenate([X_dmd_train, X_dmd_test], axis=0)

    # 7. Reshape and compute spatial means
    n_spatial = X.shape[0]
    n_time = X.shape[1]
    n_lat = len(ds.latitude.values)
    n_lon = len(ds.longitude.values)

    X_dmd = X_dmd[:n_time, :n_spatial].T

    # Compute weighted spatial means
    X_true_mean = np.average(
        np.average(
            X.reshape(n_spatial, n_time).reshape(-1, n_lat, n_lon),
            weights=np.cos(np.deg2rad(ds.latitude.values)),
            axis=1,
        ),
        axis=1,
    )
    X_dmd_mean = np.average(
        np.average(
            X_dmd.reshape(-1, n_lat, n_lon),
            weights=np.cos(np.deg2rad(ds.latitude.values)),
            axis=1,
        ),
        axis=1,
    )

    # Print DMD diagnostics
    print("\nDMD Diagnostics:")
    print(f"Shape of modes: {modes.shape}")
    print(f"Number of modes: {modes.shape[1]}")
    print(f"Shape of reconstructed data: {X_dmd.shape}")

    # Mode energies and frequencies
    print("\nMode details:")
    mode_energies = np.abs(amplitudes) * np.abs(modes).sum(axis=0)
    dt = t[1] - t[0]  # time step in hours

    for i, (energy, eig) in enumerate(zip(mode_energies, eigs, strict=False)):
        freq = np.angle(eig) / (2 * np.pi * dt)
        period = 1 / abs(freq) if freq != 0 else float("inf")
        print(f"Mode {i}:")
        print(f"  Energy: {energy:.2e}")
        print(f"  Frequency: {freq:.6f} cycles/hour " f"(period = {period:.1f} hours)")
        print(f"  Magnitude: {np.abs(eig):.6f}")

    # Print ranges
    print("\nRanges:")
    print(f"Eigenvalues: {np.min(np.abs(eigs)):.6f} " f"to {np.max(np.abs(eigs)):.6f}")
    print(
        f"Amplitudes: {np.min(np.abs(amplitudes)):.6f} "
        f"to {np.max(np.abs(amplitudes)):.6f}"
    )

    # After computing means, calculate errors using real parts only
    rmse_train = np.sqrt(
        np.mean((np.real(X_true_mean[:T_train]) - np.real(X_dmd_mean[:T_train])) ** 2)
    )
    rmse_test = np.sqrt(
        np.mean((np.real(X_true_mean[T_train:]) - np.real(X_dmd_mean[T_train:])) ** 2)
    )
    print(f"\nRMSE (training): {rmse_train:.4f} K")
    print(f"RMSE (prediction): {rmse_test:.4f} K")

    # Calculate spatial standard deviation at each timestep (using real parts)
    X_true_std = np.std(np.real(X.reshape(-1, n_lat, n_lon)), axis=(1, 2))
    X_dmd_std = np.std(np.real(X_dmd.reshape(-1, n_lat, n_lon)), axis=(1, 2))

    # Create the plot with error bands
    plt.figure(figsize=(12, 8))

    # Plot means with spatial standard deviation bands
    plt.fill_between(
        t_train,
        X_true_mean - X_true_std,
        X_true_mean + X_true_std,
        color="r",
        alpha=0.2,
        label="True variability",
    )
    plt.fill_between(
        t_train,
        X_dmd_mean - X_dmd_std,
        X_dmd_mean + X_dmd_std,
        color="grey",
        alpha=0.2,
        label="DMD variability",
    )

    # Plot means as before
    plt.plot(t_train, X_true_mean, color="r", label="True values")
    plt.plot(t_train, X_dmd_mean, color="grey", label="DMD reconstruction/prediction")
    plt.axvline(t_train[T_train], linestyle="--", color="k", label="Train/Test split")

    # Add RMSE values to plot
    plt.text(
        0.02,
        0.98,
        f"Training RMSE: {rmse_train:.4f} K",
        transform=plt.gca().transAxes,
        verticalalignment="top",
    )
    plt.text(
        0.02,
        0.94,
        f"Prediction RMSE: {rmse_test:.4f} K",
        transform=plt.gca().transAxes,
        verticalalignment="top",
    )

    plt.ylabel("Spatial mean temperature (K)")
    plt.xlabel("Hours")
    plt.legend()
    plt.title("DMD Reconstruction and Prediction with Spatial Variability")

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

    # After creating X_test
    print("\nTest Set Evaluation:")
    # Calculate reconstruction error on test set
    test_error = np.mean((X_test - X_dmd_test.T) ** 2)
    print(f"Mean squared error on test set: {test_error:.6f}")

    # Calculate relative error
    relative_error = np.linalg.norm(X_test - X_dmd_test.T) / np.linalg.norm(X_test)
    print(f"Relative error on test set: {relative_error:.6f}")

    # Add test metrics to metadata
    metadata.update(
        {"test_mse": float(test_error), "test_relative_error": float(relative_error)}
    )

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
