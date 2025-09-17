import numpy as np
from utilities import logistic_func
from tqdm import tqdm
import h5py

# Set random seed for reproducibility
np.random.seed(0)

# Load data
with h5py.File('../results/new_nat_videos_gabor_responses_full_res_z_score.h5', 'r') as all_gabor_responses:
    orientation_arr = all_gabor_responses['orientation_arr'][()]
    phase_arr = all_gabor_responses['phase_arr'][()]
    position_arr = all_gabor_responses['position_arr'][()]
    wavelength_arr = all_gabor_responses['wavelength_arr'][()]

# Constants
RESOLUTION = (1080, 1920)
FOV = (61, 92)
H_PIXELS_PER_DEGREE = RESOLUTION[1] / FOV[1]
V_PIXELS_PER_DEGREE = RESOLUTION[0] / FOV[0]
PIXELS_PER_DEGREE = int(np.ceil((H_PIXELS_PER_DEGREE + V_PIXELS_PER_DEGREE) / 2))

# Spatial frequency
freq_arr = PIXELS_PER_DEGREE / wavelength_arr
LOW_SPATIAL_FREQ_IDX = [0, 1, 2, 3]

# Load optimal parameters
WINDOW_LENGTH = 2
optimal_k_stationary = np.load(f'avg_sd_optimal_k_arr_stationary_window_length_{WINDOW_LENGTH}.npy')
optimal_L_stationary = np.load(f'avg_sd_optimal_L_arr_stationary_window_length_{WINDOW_LENGTH}.npy')
optimal_k_moving = np.load(f'avg_sd_optimal_k_arr_moving_window_length_{WINDOW_LENGTH}.npy')
optimal_L_moving = np.load(f'avg_sd_optimal_L_arr_moving_window_length_{WINDOW_LENGTH}.npy')

# Gabor filter function
def gabor_filter_func(sigma, theta, gamma, wavelength, phase, filt_size, x_offset=0, y_offset=0):
    y, x = np.meshgrid(
        np.arange(filt_size[1]) - filt_size[1] // 2,
        np.arange(filt_size[0]) - filt_size[0] // 2,
    )
    x -= x_offset
    y += y_offset
    theta_rad = np.radians(theta)
    x_prime = x * np.cos(theta_rad) + y * np.sin(theta_rad)
    y_prime = -x * np.sin(theta_rad) + y * np.cos(theta_rad)
    filter = np.exp(-0.5 * (x_prime**2 + (gamma * y_prime) ** 2) / sigma**2) * \
             np.cos(2 * np.pi * x_prime / wavelength + np.radians(phase))
    return filter / np.linalg.norm(filter)

# Gabor filter bank creation
orientation_arr = np.linspace(0, 157.5, 8)
phase_arr = np.linspace(0, 270, 4)
position_arr = np.array([[-RESOLUTION[0] // 4, RESOLUTION[1] // 4], [-RESOLUTION[0] // 4, 0],
                         [-RESOLUTION[0] // 4, -RESOLUTION[1] // 4], [0, RESOLUTION[1] // 4],
                         [0, 0], [0, -RESOLUTION[1] // 4],
                         [RESOLUTION[0] // 4, RESOLUTION[1] // 4], [RESOLUTION[0] // 4, 0],
                         [RESOLUTION[0] // 4, -RESOLUTION[1] // 4]])

filter_bank_shape = (
    len(orientation_arr), len(phase_arr), len(LOW_SPATIAL_FREQ_IDX), len(position_arr), *RESOLUTION
)
gabor_filter_bank = np.zeros(filter_bank_shape)

for i, orientation in tqdm(enumerate(orientation_arr), desc="Creating Gabor filters", total=len(orientation_arr)):
    for j, phase in enumerate(phase_arr):
        for l, wavelength in enumerate(wavelength_arr[LOW_SPATIAL_FREQ_IDX]):
            for m, position in enumerate(position_arr):
                gabor_filter_bank[i, j, l, m] = gabor_filter_func(
                    sigma=wavelength / 3,
                    theta=orientation,
                    wavelength=wavelength,
                    phase=phase,
                    gamma=1,
                    filt_size=RESOLUTION,
                    x_offset=position[0],
                    y_offset=position[1],
                )

# Contrast tuning curves
contrast_arr = np.logspace(0, 3, 50)
# append 0 to the beginning of the contrast array
contrast_arr = np.insert(contrast_arr, 0, 0)
curve_shape = (
    len(orientation_arr), len(phase_arr), len(LOW_SPATIAL_FREQ_IDX), len(position_arr), len(contrast_arr)
)
contrast_tuning_curves_stationary = np.zeros(curve_shape)
contrast_tuning_curves_moving = np.zeros(curve_shape)

lambda_idx = 4
lambda_arr = np.arange(0, 2.25, 0.25)

for i, orientation in tqdm(enumerate(orientation_arr), desc="Computing tuning curves", total=len(orientation_arr)):
    for j, phase in enumerate(phase_arr):
        for l, wavelength_idx in enumerate(LOW_SPATIAL_FREQ_IDX):
            for m, position in enumerate(position_arr):
                sample_filter = gabor_filter_bank[i, j, l, m]

                k_stat = optimal_k_stationary[i, j, l, m, lambda_idx]
                L_stat = optimal_L_stationary[i, j, l, m, lambda_idx]
                k_mov = optimal_k_moving[i, j, l, m, lambda_idx]
                L_mov = optimal_L_moving[i, j, l, m, lambda_idx]

                filter_outputs = contrast_arr[:, None, None] * sample_filter
                filter_responses = np.sum(sample_filter * filter_outputs, axis=(-2, -1))

                stationary = logistic_func(filter_responses, k=k_stat, L=L_stat)
                moving = logistic_func(filter_responses, k=k_mov, L=L_mov)

                norm_factor = max(np.max(stationary), np.max(moving))
                contrast_tuning_curves_stationary[i, j, l, m] = stationary / norm_factor
                contrast_tuning_curves_moving[i, j, l, m] = moving / norm_factor

# Save results
np.save(f'contrast_tuning_curves_stationary_window_length_{WINDOW_LENGTH}_lambda_{lambda_arr[lambda_idx]}.npy', contrast_tuning_curves_stationary)
np.save(f'contrast_tuning_curves_moving_window_length_{WINDOW_LENGTH}_lambda_{lambda_arr[lambda_idx]}.npy', contrast_tuning_curves_moving)