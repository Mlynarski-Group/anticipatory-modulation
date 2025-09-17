import numpy as np
from utilities import logistic_func
from tqdm import tqdm
import h5py
import bottleneck as bn
from joblib import Parallel, delayed
from sklearn.linear_model import LinearRegression

# --- Constants and Parameters ---
RESOLUTION = (1080, 1920)
FOV = (61, 92)
H_PIXELS_PER_DEGREE = RESOLUTION[1] / FOV[1]
V_PIXELS_PER_DEGREE = RESOLUTION[0] / FOV[0]
PIXELS_PER_DEGREE = int(np.ceil((H_PIXELS_PER_DEGREE + V_PIXELS_PER_DEGREE) / 2))
WINDOW_LENGTH = 5
LOW_SPATIAL_FREQ_IDX = np.arange(0, 31)
CONTRAST_ARR = [10, 20, 50, 100]
GRATING_ORIENTATION_ARR = np.linspace(0, 360, 16, endpoint=False)
LAMBDA_IDX = 4
LAMBDA_ARR = np.arange(0, 2.25, .25)

np.random.seed(0)

def compute_optimal_k_L():
    '''
    This function optimizes logistic nonlinearities for a variety of Gaussian distributions
    and saves the optimal parameters for later use.
    '''
    all_gabor_responses = h5py.File('../results/new_nat_videos_gabor_responses_full_res_z_score_more_freq.h5', 'r')

    resolution_height = 1080
    resolution_width = 1920
    horizontal_fov = 92
    vertical_fov = 61

    horizontal_pixels_per_degree = resolution_width / horizontal_fov
    vertical_pixels_per_degree = resolution_height / vertical_fov
    pixels_per_degree = np.ceil((horizontal_pixels_per_degree + vertical_pixels_per_degree) / 2)
    print(pixels_per_degree)

    orientation_arr = all_gabor_responses['orientation_arr'][()]
    phase_arr = all_gabor_responses['phase_arr'][()]
    position_arr = all_gabor_responses['position_arr'][()]
    wavelength_arr = all_gabor_responses['wavelength_arr'][()]
    freq_arr = pixels_per_degree / wavelength_arr
    print(freq_arr)
    environments = ['field', 'forest', 'orchard', 'tall_grass', 'pond']
    fps = 30
    window_size = WINDOW_LENGTH * fps
    num_samples = 1740

    all_gabor_data = {}
    for env_key in tqdm(environments, desc="Pre-loading environments"):
        env_data = {}
        for vid_key in tqdm(all_gabor_responses[env_key].keys(), desc=f"Loading {env_key}", leave=False):
            env_data[vid_key] = all_gabor_responses[env_key][vid_key][()]
        all_gabor_data[env_key] = env_data

    def compute_sd_for_video(data, window, idx, num_samples):
        return bn.move_std(data, window=window, min_count=window, axis=-1)[:, :, idx, :, :num_samples]

    stationary_stim_SD = []
    moving_stim_SD = []

    for env_key in tqdm(environments, desc="Environments", leave=False):
        env_data = all_gabor_data[env_key]
        vid_keys = list(env_data.keys())
        results = Parallel(n_jobs=-1)(
            delayed(compute_sd_for_video)(env_data[vid_key], window_size, LOW_SPATIAL_FREQ_IDX, num_samples)
            for vid_key in tqdm(vid_keys, desc=f"Processing {env_key}", leave=False)
        )
        for vid_key, resp_SD in zip(vid_keys, results):
            if 'stationary' in vid_key:
                stationary_stim_SD.append(resp_SD)
            if 'moving' in vid_key and 'free_moving' not in vid_key:
                moving_stim_SD.append(resp_SD)

    stationary_stim_SD = np.array(stationary_stim_SD)
    moving_stim_SD = np.array(moving_stim_SD)

    windowed_std_stationary_responses = np.nanmean(stationary_stim_SD, axis=(0, -1))
    windowed_std_moving_responses = np.nanmean(moving_stim_SD, axis=(0, -1))

    print(windowed_std_stationary_responses.shape)
    print(windowed_std_moving_responses.shape)

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

    optimal_k_arr_moving = np.zeros((len(orientation_arr), len(phase_arr), len(wavelength_arr[LOW_SPATIAL_FREQ_IDX]), len(position_arr), len(lambda_arr)))
    optimal_L_arr_moving = np.zeros((len(orientation_arr), len(phase_arr), len(wavelength_arr[LOW_SPATIAL_FREQ_IDX]), len(position_arr), len(lambda_arr)))
    optimal_k_arr_stationary = np.zeros((len(orientation_arr), len(phase_arr), len(wavelength_arr[LOW_SPATIAL_FREQ_IDX]), len(position_arr), len(lambda_arr)))
    optimal_L_arr_stationary = np.zeros((len(orientation_arr), len(phase_arr), len(wavelength_arr[LOW_SPATIAL_FREQ_IDX]), len(position_arr), len(lambda_arr)))

    for i, orientation in tqdm(enumerate(orientation_arr)):
        print("orientation: " + str(orientation))
        for j, phase in enumerate(phase_arr):
            for l, wavelength in enumerate(wavelength_arr[LOW_SPATIAL_FREQ_IDX]):
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

    np.save(f'avg_sd_optimal_k_arr_moving_window_length_{WINDOW_LENGTH}.npy', optimal_k_arr_moving)
    np.save(f'avg_sd_optimal_L_arr_moving_window_length_{WINDOW_LENGTH}.npy', optimal_L_arr_moving)
    np.save(f'avg_sd_optimal_k_arr_stationary_window_length_{WINDOW_LENGTH}.npy', optimal_k_arr_stationary)
    np.save(f'avg_sd_optimal_L_arr_stationary_window_length_{WINDOW_LENGTH}.npy', optimal_L_arr_stationary)

def load_data():
    with h5py.File('../results/new_nat_videos_gabor_responses_full_res_z_score_more_freq.h5', 'r') as data:
        orientation_arr = data['orientation_arr'][()]
        phase_arr = data['phase_arr'][()]
        position_arr = data['position_arr'][()]
        wavelength_arr = data['wavelength_arr'][()]
    optimal_k_stationary = np.load(f'avg_sd_optimal_k_arr_stationary_window_length_{WINDOW_LENGTH}.npy')
    optimal_L_stationary = np.load(f'avg_sd_optimal_L_arr_stationary_window_length_{WINDOW_LENGTH}.npy')
    optimal_k_moving = np.load(f'avg_sd_optimal_k_arr_moving_window_length_{WINDOW_LENGTH}.npy')
    optimal_L_moving = np.load(f'avg_sd_optimal_L_arr_moving_window_length_{WINDOW_LENGTH}.npy')
    return orientation_arr, phase_arr, position_arr, wavelength_arr, optimal_k_stationary, optimal_L_stationary, optimal_k_moving, optimal_L_moving

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
    filt = np.exp(-0.5 * (x_prime**2 + (gamma * y_prime) ** 2) / sigma**2) * \
           np.cos(2 * np.pi * x_prime / wavelength + np.radians(phase))
    return filt / np.linalg.norm(filt)

def generate_filter_inputs(grating_orientations, wavelength, phase, position, filt_size):
    return np.array([
        gabor_filter_func(
            sigma=wavelength / 3,
            theta=orientation,
            wavelength=wavelength,
            phase=phase,
            gamma=1,
            filt_size=filt_size,
            x_offset=position[0],
            y_offset=position[1],
        ) for orientation in grating_orientations
    ])

def compute_ori_tuning():
    orientation_arr, phase_arr, position_arr, wavelength_arr, \
    optimal_k_stationary, optimal_L_stationary, optimal_k_moving, optimal_L_moving = load_data()

    filter_bank_shape = (
        len(CONTRAST_ARR), len(orientation_arr), len(phase_arr), len(LOW_SPATIAL_FREQ_IDX), len(position_arr), len(GRATING_ORIENTATION_ARR)
    )
    additive_gains = np.zeros(filter_bank_shape[:-1])
    multiplicative_gains = np.zeros(filter_bank_shape[:-1])
    stationary_tuning_curves = np.zeros(filter_bank_shape)
    moving_tuning_curves = np.zeros(filter_bank_shape)

    for contrast_idx, contrast_param in tqdm(enumerate(CONTRAST_ARR), desc="Processing contrast"):
        for orientation_idx, orientation in tqdm(enumerate(orientation_arr), desc="Processing orientations"):
            for phase_idx, phase in enumerate(phase_arr):
                for wavelength_idx in LOW_SPATIAL_FREQ_IDX:
                    wavelength = wavelength_arr[wavelength_idx]
                    for position_idx, position in enumerate(position_arr):
                        sample_filter = gabor_filter_func(
                            sigma=wavelength / 3,
                            theta=orientation,
                            wavelength=wavelength,
                            phase=phase,
                            gamma=1,
                            filt_size=RESOLUTION,
                            x_offset=position[0],
                            y_offset=position[1],
                        )
                        filter_inputs = generate_filter_inputs(GRATING_ORIENTATION_ARR, wavelength, phase, position, RESOLUTION)
                        filter_output = np.tensordot(sample_filter, contrast_param * filter_inputs, axes=([-2, -1], [-2, -1]))

                        k_stationary = optimal_k_stationary[orientation_idx, phase_idx, wavelength_idx, position_idx, LAMBDA_IDX]
                        L_stationary = optimal_L_stationary[orientation_idx, phase_idx, wavelength_idx, position_idx, LAMBDA_IDX]
                        k_moving = optimal_k_moving[orientation_idx, phase_idx, wavelength_idx, position_idx, LAMBDA_IDX]
                        L_moving = optimal_L_moving[orientation_idx, phase_idx, wavelength_idx, position_idx, LAMBDA_IDX]

                        ori_tuning_stationary = logistic_func(filter_output, k=k_stationary, L=L_stationary)
                        ori_tuning_moving = logistic_func(filter_output, k=k_moving, L=L_moving)

                        reg = LinearRegression().fit(ori_tuning_stationary.reshape(-1, 1), ori_tuning_moving)
                        additive_gains[contrast_idx, orientation_idx, phase_idx, wavelength_idx, position_idx] = reg.intercept_
                        multiplicative_gains[contrast_idx, orientation_idx, phase_idx, wavelength_idx, position_idx] = reg.coef_[0]
                        stationary_tuning_curves[contrast_idx, orientation_idx, phase_idx, wavelength_idx, position_idx] = ori_tuning_stationary
                        moving_tuning_curves[contrast_idx, orientation_idx, phase_idx, wavelength_idx, position_idx] = ori_tuning_moving

    np.save(f'orientation_additive_gains_window_length_{WINDOW_LENGTH}_lambda_{LAMBDA_ARR[LAMBDA_IDX]}.npy', additive_gains)
    np.save(f'orientation_multiplicative_gains_window_length_{WINDOW_LENGTH}_lambda_{LAMBDA_ARR[LAMBDA_IDX]}.npy', multiplicative_gains)
    np.save(f'orientation_stationary_tuning_curves_window_length_{WINDOW_LENGTH}_lambda_{LAMBDA_ARR[LAMBDA_IDX]}.npy', stationary_tuning_curves)
    np.save(f'orientation_moving_tuning_curves_window_length_{WINDOW_LENGTH}_lambda_{LAMBDA_ARR[LAMBDA_IDX]}.npy', moving_tuning_curves)
    np.save('contrast_arr.npy', CONTRAST_ARR)

if __name__ == "__main__":
    compute_optimal_k_L()
    compute_ori_tuning()