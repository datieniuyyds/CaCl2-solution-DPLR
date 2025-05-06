import MDAnalysis as mda
import numpy as np
from concurrent.futures import ProcessPoolExecutor
import time

def calculate_msd_for_lag_time(args):
    """Calculate MSD for a given lag time (for O atoms)."""
    lag_time, positions = args
    if lag_time == 0:
        return np.zeros(positions.shape[0])
    displacements = positions[:-lag_time] - positions[lag_time:]
    squared_displacements = np.square(displacements).sum(axis=2)
    msd_lag = squared_displacements.mean(axis=1)
    return msd_lag

def calculate_msd_for_lag_time_ca(args):
    """Calculate MSD for a given lag time (for Ca atoms)."""
    lag_time, positions = args
    if lag_time == 0:
        return np.zeros(positions.shape[0])
    displacements = positions[:-lag_time] - positions[lag_time:]
    squared_displacements = np.square(displacements).sum(axis=2)
    msd_lag = squared_displacements.mean(axis=1)
    return msd_lag

def calculate_msd_for_lag_time_cl(args):
    """Calculate MSD for a given lag time (for Cl atoms)."""
    lag_time, positions = args
    if lag_time == 0:
        return np.zeros(positions.shape[0])
    displacements = positions[:-lag_time] - positions[lag_time:]
    squared_displacements = np.square(displacements).sum(axis=2)
    msd_lag = squared_displacements.mean(axis=1)
    return msd_lag

start_time_loading = time.time()

print('Loading trajectory...')
u = mda.Universe("traj.pdb","traj-unwrap.xtc")

loading_time = time.time() - start_time_loading
print(f"Loading trajectory took {loading_time:.2f} seconds.")

start_time_msd = time.time()

print('MSD calculating...')
water = u.select_atoms('name O')
ca_atoms = u.select_atoms('name Ca')
cl_atoms = u.select_atoms('name Cl')
timestep = 0.5  # 0.5 fs converted to ps (0.5e-3 ps)

# Define frame range
start_frame = 1000
end_frame = 4000

# Collect positions for selected frames
water_positions = np.array([ts.positions[water.indices] for ts in u.trajectory[start_frame:end_frame]])
ca_positions = np.array([ts.positions[ca_atoms.indices] for ts in u.trajectory[start_frame:end_frame]])
cl_positions = np.array([ts.positions[cl_atoms.indices] for ts in u.trajectory[start_frame:end_frame]])

# Adjust lag times to fit the frame range (3000 frames: 1000 to 4000)
max_lag_frames = end_frame - start_frame  # 3000 frames
lag_times = np.arange(1, max_lag_frames + 1)  # 1 to 3000

max_workers = 20  # Number of processes
args = [(lag_time, water_positions) for lag_time in lag_times]
args_ca = [(lag_time, ca_positions) for lag_time in lag_times]
args_cl = [(lag_time, cl_positions) for lag_time in lag_times]

# Calculate MSD for O, Ca, and Cl atoms
with ProcessPoolExecutor(max_workers=max_workers) as executor:
    results = list(executor.map(calculate_msd_for_lag_time, args))

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    results_ca = list(executor.map(calculate_msd_for_lag_time_ca, args_ca))

with ProcessPoolExecutor(max_workers=max_workers) as executor:
    results_cl = list(executor.map(calculate_msd_for_lag_time_cl, args_cl))

# Compute mean and standard deviation of MSD
msd_mean = [np.mean(result) for result in results]
msd_std = [np.std(result) for result in results]
msd_mean_ca = [np.mean(result) for result in results_ca]
msd_std_ca = [np.std(result) for result in results_ca]
msd_mean_cl = [np.mean(result) for result in results_cl]
msd_std_cl = [np.std(result) for result in results_cl]

# Save results to text files
with open('msd_mean_new.txt', 'w') as f:
    for val in msd_mean:
        f.write(str(val) + '\n')
with open('msd_std_new.txt', 'w') as f:
    for val in msd_std:
        f.write(str(val) + '\n')
with open('msd_mean_ca.txt', 'w') as f:
    for val in msd_mean_ca:
        f.write(str(val) + '\n')
with open('msd_std_ca.txt', 'w') as f:
    for val in msd_std_ca:
        f.write(str(val) + '\n')
with open('msd_mean_cl.txt', 'w') as f:
    for val in msd_mean_cl:
        f.write(str(val) + '\n')
with open('msd_std_cl.txt', 'w') as f:
    for val in msd_std_cl:
        f.write(str(val) + '\n')

msd_calculation_time = time.time() - start_time_msd
print(f"MSD calculation took {msd_calculation_time:.2f} seconds.")

# Optionally print to console for verification
print("MSD Mean (O atoms):", msd_mean[:5], "...")  # First 5 values
print("MSD Mean (Ca atoms):", msd_mean_ca[:5], "...")
print("MSD Mean (Cl atoms):", msd_mean_cl[:5], "...")
