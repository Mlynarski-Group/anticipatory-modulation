'''
This script optimizes logistic nonlinearities for a variety of Gaussian distributions and plots the optimal parameters.
Author: Jonathan Gant
Date: 29.08.2024
Optimized for performance
'''

import numpy as np
from joblib import Parallel, delayed
from tqdm import tqdm
from scipy.stats import norm
import os
import sys

n_proc = int(os.cpu_count())

# Define the logistic non-linearity
def logistic_func(s, s_0=0, k=1, L=1):
    return L / (1 + np.exp(-k * (s - s_0)))

# make results directory if it doesn't exist
os.makedirs('../results/', exist_ok=True)

# Directory for saving results
dir_name = '../results/gaussian_optimization_analytic_fast'
os.makedirs(dir_name, exist_ok=True)

# Set random seed for reproducibility
rng = np.random.default_rng(seed=0)

# Define stimulus bins and sigma values
sigma_arr = np.logspace(-1, 3, int(10*28))  # All sigma
num_stim_bins = int(1e5)

# Define k and L values for grid search
k_arr = np.logspace(-3, 0, 200)
L_arr = np.arange(.05, 10.05, .05)

# Define response and stimulus bins
stimulus_bins = np.linspace(-3*np.max(sigma_arr), 3*np.max(sigma_arr), num_stim_bins + 1)
stimulus_bins_val = (stimulus_bins[1:] + stimulus_bins[:-1]) / 2
response_bins = np.linspace(0, np.max(L_arr), num_stim_bins + 1)
response_bins_midpoint = (response_bins[1:] + response_bins[:-1]) / 2

# for all k and L map the stimulus to the response bins
def map_stimulus_to_response(stimulus, k, L, response_bins, response_bins_midpoint):
    responses = logistic_func(stimulus, k=k, L=L)
    # digitize responses
    response_digitized_idx = np.clip(np.digitize(responses, response_bins) - 1, 0, len(response_bins_midpoint) - 1)
    responses_digitized = response_bins_midpoint[response_digitized_idx]
    return responses_digitized


# Function to compute average response and MI
def compute_quantities(stimuli_probability, k_arr, L_arr, response_bins, response_bins_midpoint, mapped_response_bins):
    """Compute average response and MI for a given stimulus."""
    n_k, n_L = len(k_arr), len(L_arr)
    average_response = np.zeros((n_k, n_L))
    MI = np.zeros((n_k, n_L))

    nonzero_entry = stimuli_probability > 0
    stim_entropy = -np.sum(stimuli_probability[nonzero_entry] * np.log2(stimuli_probability[nonzero_entry]))

    for i, k in enumerate(k_arr):
        for j, L in enumerate(L_arr):
            response_counts = np.bincount(np.digitize(mapped_response_bins[i, j, :], response_bins) - 1, weights=stimuli_probability, minlength=len(response_bins) - 1)
            response_probability = response_counts / np.sum(response_counts)
            
            # compute average response
            average_response[i, j] = np.dot(response_probability, response_bins_midpoint)
            
            nonzero_entry = response_probability > 0  # Mask for non-zero probabilities
            MI[i, j] = -np.sum(response_probability[nonzero_entry] * np.log2(response_probability[nonzero_entry]))
    
    return average_response, MI, stim_entropy


# map the stimulus bins to the response bins for all k and L
mapped_response_bins = np.zeros((len(k_arr), len(L_arr), len(stimulus_bins_val)))
print("Mapping stimulus to response bins...")
# loop over k and L
for i, k in tqdm(enumerate(k_arr)):
    for j, L in enumerate(L_arr):
        mapped_response_bins[i, j, :] = map_stimulus_to_response(stimulus_bins_val, k, L, response_bins, response_bins_midpoint)

# compute the stimuli probability for each sigma
stimuli_probability_arr = np.zeros((len(sigma_arr), len(stimulus_bins_val)))
print("Computing stimulus probabilities...")
for m, sigma in tqdm(enumerate(sigma_arr)):
    stimuli_probability = norm.pdf(stimulus_bins_val, scale=sigma)
    stimuli_probability /= np.sum(stimuli_probability)
    stimuli_probability_arr[m] = stimuli_probability

# Parallel computation of quantities for all stimuli
results = Parallel(n_jobs=n_proc, backend="loky")(
    delayed(compute_quantities)(stimuli_probability_arr[m, :], k_arr, L_arr, response_bins, response_bins_midpoint, mapped_response_bins)
    for m in tqdm(range(len(sigma_arr)))
)

# Extract and save results
average_response_arr = np.array([res[0] for res in results])
MI_arr = np.array([res[1] for res in results])
stim_entropy_arr = np.array([res[2] for res in results])

np.save(os.path.join(dir_name, 'MI_arr.npy'), MI_arr)
np.save(os.path.join(dir_name, 'average_response_arr.npy'), average_response_arr)
np.save(os.path.join(dir_name, 'stim_entropy_arr.npy'), stim_entropy_arr)
np.save(os.path.join(dir_name, 'sigma_arr.npy'), sigma_arr)
np.save(os.path.join(dir_name, 'k_arr.npy'), k_arr)
np.save(os.path.join(dir_name, 'L_arr.npy'), L_arr)
np.save(os.path.join(dir_name, 'response_bins.npy'), response_bins)

print("Optimization complete! Results saved.")