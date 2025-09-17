'''
This script optimizes logistic nonlinearities for a variety of Gaussian distributions and plots the optimal parameters.
Author: Jonathan Gant
Date: 29.08.2024
'''

import numpy as np
import matplotlib.pyplot as plt
from utilities import logistic_func, calc_MI, calc_entropy
from tqdm import tqdm
import h5py
import bottleneck as bn
from joblib import Parallel, delayed
from scipy.ndimage import gaussian_filter

# Set random seed for reproducibility
np.random.seed(0)

# Load in the data
all_gabor_responses = h5py.File('../results/new_nat_videos_gabor_responses_full_res_z_score_more_freq.h5', 'r')

# Video size and FOV
resolution_height = 1080
resolution_width = 1920
horizontal_fov = 92
vertical_fov = 61

# Conversion factor of pixels to degrees
horizontal_pixels_per_degree = resolution_width / horizontal_fov
vertical_pixels_per_degree = resolution_height / vertical_fov
pixels_per_degree = np.ceil((horizontal_pixels_per_degree + vertical_pixels_per_degree) / 2)
print(pixels_per_degree)

# Data hyperparameters
orientation_arr = all_gabor_responses['orientation_arr'][()]
phase_arr = all_gabor_responses['phase_arr'][()]
position_arr = all_gabor_responses['position_arr'][()]
wavelength_arr = all_gabor_responses['wavelength_arr'][()]
freq_arr = pixels_per_degree / wavelength_arr
print(freq_arr)
low_spatial_freq_idx = np.arange(0, 31)
high_spatial_freq_idx = np.arange(35, 70)

environments = ['field', 'forest', 'orchard', 'tall_grass', 'pond']
fps = 30
window_length = 5 # seconds
window_size = window_length * fps
num_samples = 1740

# Pre-load all data into memory for faster access
all_gabor_data = {}
for env_key in tqdm(environments, desc="Pre-loading environments"):
    env_data = {}
    for vid_key in tqdm(all_gabor_responses[env_key].keys(), desc=f"Loading {env_key}", leave=False):
        env_data[vid_key] = all_gabor_responses[env_key][vid_key][()]
    all_gabor_data[env_key] = env_data

def compute_sd_for_video(data, window, idx, num_samples):
    # data: shape (..., time)
    return bn.move_std(data, window=window, min_count=window, axis=-1)[:, :, idx, :, :num_samples]

stationary_stim_SD = []
moving_stim_SD = []

for env_key in tqdm(environments, desc="Environments", leave=False):
    env_data = all_gabor_data[env_key]
    vid_keys = list(env_data.keys())
    results = Parallel(n_jobs=-1)(
        delayed(compute_sd_for_video)(env_data[vid_key], window_size, low_spatial_freq_idx, num_samples)
        for vid_key in tqdm(vid_keys, desc=f"Processing {env_key}", leave=False)
    )
    for vid_key, resp_SD in zip(vid_keys, results):
        if 'stationary' in vid_key:
            stationary_stim_SD.append(resp_SD)
        if 'moving' in vid_key and 'free_moving' not in vid_key:
            moving_stim_SD.append(resp_SD)

# Make arrays
stationary_stim_SD = np.array(stationary_stim_SD)
moving_stim_SD = np.array(moving_stim_SD)

windowed_std_stationary_responses = np.nanmean(stationary_stim_SD, axis=(0, -1))
windowed_std_moving_responses = np.nanmean(moving_stim_SD, axis=(0, -1))

print(windowed_std_stationary_responses.shape)
print(windowed_std_moving_responses.shape)

# Optimize nonlinearities using lookup table
dir_name = 'gaussian_optimization_analytic_fast'
MI_arr = np.load(dir_name + '/MI_arr.npy')
average_response_arr = np.load(dir_name + '/average_response_arr.npy')
stimulus_entropy = np.load(dir_name + '/stim_entropy_arr.npy')
response_bins = np.load(dir_name + '/response_bins.npy')
k_arr = np.load(dir_name + '/k_arr.npy')
L_arr = np.load(dir_name + '/L_arr.npy')
sigma_arr = np.load(dir_name + '/sigma_arr.npy')

lambda_arr = np.arange(0, 10.5, .5)
optimal_k = np.zeros((len(sigma_arr), len(lambda_arr)))
optimal_L = np.zeros((len(sigma_arr), len(lambda_arr)))
optimal_k_idx = np.zeros((len(sigma_arr), len(lambda_arr)), dtype=np.int32)
optimal_L_idx = np.zeros((len(sigma_arr), len(lambda_arr)), dtype=np.int32)

for i, lambda_ in enumerate(lambda_arr):
    utility = MI_arr - lambda_ * average_response_arr
    print(utility.shape)
    for j, sigma in enumerate(sigma_arr):
        idx = np.unravel_index(np.argmax(utility[j, :, :]), utility[j, :, :].shape)
        optimal_k_idx[j, i] = int(idx[0])
        optimal_L_idx[j, i] = int(idx[1])
        optimal_k[j, i] = k_arr[optimal_k_idx[j, i]]
        optimal_L[j, i] = L_arr[optimal_L_idx[j, i]]

# Loop over all orientations, phases, wavelengths and compute the optimal parameters for each
optimal_k_arr_moving = np.zeros((len(orientation_arr), len(phase_arr), len(wavelength_arr[low_spatial_freq_idx]), len(position_arr), len(lambda_arr)))
optimal_L_arr_moving = np.zeros((len(orientation_arr), len(phase_arr), len(wavelength_arr[low_spatial_freq_idx]), len(position_arr), len(lambda_arr)))
optimal_k_arr_stationary = np.zeros((len(orientation_arr), len(phase_arr), len(wavelength_arr[low_spatial_freq_idx]), len(position_arr), len(lambda_arr)))
optimal_L_arr_stationary = np.zeros((len(orientation_arr), len(phase_arr), len(wavelength_arr[low_spatial_freq_idx]), len(position_arr), len(lambda_arr)))

for i, orientation in tqdm(enumerate(orientation_arr)):
    print("orientation: " + str(orientation))
    for j, phase in enumerate(phase_arr):
        for l, wavelength in enumerate(wavelength_arr[low_spatial_freq_idx]):
            for m, position in enumerate(position_arr):
                test_moving_std = windowed_std_moving_responses[i, j, l, m]
                test_stationary_std = windowed_std_stationary_responses[i, j, l, m]
                for n, lambda_ in enumerate(lambda_arr):
                    moving_idx = np.argmin(np.abs(sigma_arr - test_moving_std))
                    optimal_k_arr_moving[i, j, l, m, n] = optimal_k[moving_idx, n]
                    optimal_L_arr_moving[i, j, l, m, n] = optimal_L[moving_idx, n]
                    stationary_idx = np.argmin(np.abs(sigma_arr - test_stationary_std))
                    optimal_k_arr_stationary[i, j, l, m, n] = optimal_k[stationary_idx, n]
                    optimal_L_arr_stationary[i, j, l, m, n] = optimal_L[stationary_idx, n]

np.save(f'avg_sd_optimal_k_arr_moving_window_length_{window_length}.npy', optimal_k_arr_moving)
np.save(f'avg_sd_optimal_L_arr_moving_window_length_{window_length}.npy', optimal_L_arr_moving)
np.save(f'avg_sd_optimal_k_arr_stationary_window_length_{window_length}.npy', optimal_k_arr_stationary)
np.save(f'avg_sd_optimal_L_arr_stationary_window_length_{window_length}.npy', optimal_L_arr_stationary)