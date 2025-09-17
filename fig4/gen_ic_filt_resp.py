'''
This script generates the responses of IC filters to normalized natural image patches

Author: Jonathan Gant
Date: 29.01.2025
'''

import numpy as np
from decord import VideoReader, cpu
import glob
import os
import cv2
from tqdm import tqdm
from sklearn.decomposition import FastICA

data_group = 'nat_videos'

# define parameters of the video
vid_data_dir = "../data/" + data_group + "/"

resolution_height = 1080 // 8
resolution_width = 1920 // 8

num_distances = 4

# load in the video data
def load_video(fname, patch_size=32, num_patches_per_video=1024, random_sampling=True, num_distances=4):
    vr = VideoReader(fname, ctx=cpu(0))

    if random_sampling:
        # randomly select frames with replacement
        frame_idx_list = np.random.choice(len(vr), num_patches_per_video, replace=True)
        patch_arr = np.zeros((num_patches_per_video, patch_size**2))
    else:
        frame_idx_list = np.arange(len(vr))
        patch_arr = np.zeros((len(vr), num_distances, patch_size**2))
        total_distance = resolution_width//2-patch_size//2
        step_size = total_distance // num_distances

    # loop over each frame and select a random image patch and normalize it
    for i, idx in enumerate(frame_idx_list):
        frame = vr[idx].asnumpy()
        grayscale_frame = np.mean(frame, axis=-1)
        downscale_frame = cv2.resize(grayscale_frame, (resolution_width, resolution_height), interpolation=cv2.INTER_AREA)
        if random_sampling:
            while True:
                x = np.random.randint(0, downscale_frame.shape[0] - patch_size)
                y = np.random.randint(0, downscale_frame.shape[1] - patch_size)
                patch = downscale_frame[x:x+patch_size, y:y+patch_size].flatten()
                patch_sd = np.std(patch)
                if patch_sd > 0:
                    patch_arr[i, :] = (patch - np.mean(patch)) / patch_sd
                    break
        else:
            for j in range(num_distances):
                patch = downscale_frame[downscale_frame.shape[0]//2-patch_size//2:downscale_frame.shape[0]//2+patch_size//2, downscale_frame.shape[1]//2-patch_size//2+j*step_size:downscale_frame.shape[1]//2+patch_size//2+j*step_size].flatten()
                assert len(patch) == patch_size**2
                patch_sd = np.std(patch)
                if patch_sd > 0:
                    patch_arr[i, j, :] = (patch - np.mean(patch)) / patch_sd
                else:
                    # add small amount of noise and then normalize
                    print("Warning: zero variance patch. Adding noise...")
                    patch += np.random.normal(patch_size**2)
                    patch_arr[i, j, :] = (patch - np.mean(patch)) / np.std(patch)

    return patch_arr

# data_dir = 'ICA_less_data_more_iter' # 1024 patches per video, 20000 iterations
data_dir = 'PCA_then_ICA' # 1024 patches per video, 20000 iterations

# load IC filters
print("Loading IC components...")
# ic_components = np.load(data_dir + "/ica_components_picard_tol_1e-12_unit_var.npy")
ic_components = np.load(data_dir + "/ica_components_FastICA.npy")

# generate the response to the randomly sampled data
# loading randomly sampled image patches
print("Loading image patches...")
random_image_patches = np.load(data_dir + "/image_patches.npy")
print("Collecting patches based on stationary and moving conditions...")

patch_size = 32
num_patches_per_video = 5000

# every 10 videos, the environment changes, so every 50000 samples the environment changes
chunk_size = 50000

random_stationary_image_patches = []
random_moving_image_patches = []
for i in range(0, len(random_image_patches), 2*chunk_size):
    random_moving_image_patches.append(random_image_patches[i:i+chunk_size, :])
    random_stationary_image_patches.append(random_image_patches[i+chunk_size:i+2*chunk_size, :])

random_stationary_image_patches = np.vstack(random_stationary_image_patches)
random_moving_image_patches = np.vstack(random_moving_image_patches)

# generate responses
random_stationary_responses = ic_components @ random_stationary_image_patches.T
random_moving_responses = ic_components @ random_moving_image_patches.T

np.save(data_dir + "/stationary_responses_random_more_data_FastICA.npy", random_stationary_responses)
np.save(data_dir + "/moving_responses_random_more_data_FastICA.npy", random_moving_responses)

# load in the video data for the preserved spatiotemporal sequence

# parallelize the loading of the video data
import multiprocessing
from joblib import Parallel, delayed
print("Loading video filenames...")

environments = ["field", "forest", "orchard", "tall_grass", "pond"]
patch_size = 32
num_patches_per_video = 1500
num_videos = int(2*len(environments)*10)

stationary_fnames = []
moving_fnames = []
for environment_name in environments:
    for fname in glob.glob(vid_data_dir + environment_name + "/*.MP4"):
        if "stationary" in fname:
            stationary_fnames.append(fname)
        elif "free_moving" not in fname:
            moving_fnames.append(fname)

num_cores = multiprocessing.cpu_count()
print("using " + str(num_cores) + " cores")
# load_video(fname, patch_size=patch_size, num_patches_per_video=num_patches_per_video)

# check if the data exists already
if os.path.exists(data_dir + "/stationary_image_patches_nonrandom_multi_dist.npy") and os.path.exists(data_dir + "/moving_image_patches_nonrandom_multi_dist.npy"):
    print("Image patches already exist. Loading...")
    stationary_image_patches = np.load(data_dir + "/stationary_image_patches_nonrandom_multi_dist.npy")
    moving_image_patches = np.load(data_dir + "/moving_image_patches_nonrandom_multi_dist.npy")
    print(f"Collected patches shape: {np.shape(stationary_image_patches)}")
    print("Skipping to generating responses")
else:
    stationary_image_patches = Parallel(n_jobs=num_cores)(delayed(load_video)(fname, patch_size=patch_size, num_patches_per_video=num_patches_per_video, random_sampling=False) for fname in stationary_fnames)
    moving_image_patches = Parallel(n_jobs=num_cores)(delayed(load_video)(fname, patch_size=patch_size, num_patches_per_video=num_patches_per_video, random_sampling=False) for fname in moving_fnames)

    stationary_image_patches = np.vstack(stationary_image_patches)
    moving_image_patches = np.vstack(moving_image_patches)

    # save the image patches
    np.save(data_dir + "/stationary_image_patches_nonrandom_multi_dist.npy", stationary_image_patches)
    np.save(data_dir + "/moving_image_patches_nonrandom_multi_dist.npy", moving_image_patches)

print(np.shape(stationary_image_patches))
print(np.shape(moving_image_patches))

# generate responses
stationary_responses = np.zeros((ic_components.shape[0], stationary_image_patches.shape[1], stationary_image_patches.shape[0]))
moving_responses = np.zeros((ic_components.shape[0], moving_image_patches.shape[1], moving_image_patches.shape[0]))

# multiply with the patches
for j in range(stationary_image_patches.shape[1]):
    stationary_responses[:, j, :] = ic_components @ stationary_image_patches[:, j, :].T
    moving_responses[:, j, :] = ic_components @ moving_image_patches[:, j, :].T

# save the mixing and unmixing matrices
np.save(data_dir + "/stationary_responses_nonrandom_multi_dist_FastICA.npy", stationary_responses)
np.save(data_dir + "/moving_responses_nonrandom_multi_dist_FastICA.npy", moving_responses)
# np.save(data_dir + "/stationary_responses_nonrandom_multi_dist_tol_1e-12_unit_var.npy", stationary_responses)
# np.save(data_dir + "/moving_responses_nonrandom_multi_dist_tol_1e-12_unit_var.npy", moving_responses)