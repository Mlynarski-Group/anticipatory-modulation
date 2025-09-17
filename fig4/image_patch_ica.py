'''
This script implements ICA on image patches from natural videos. 

Author: Jonathan Gant
Date: 29.01.2025
'''

import numpy as np
from decord import VideoReader, cpu
import glob
import os
import cv2
import multiprocessing
from joblib import Parallel, delayed
from sklearn.decomposition import FastICA
from sklearn.decomposition import PCA

data_group = 'nat_videos'

# define parameters of the video
vid_data_dir = "../data/" + data_group + "/"

resolution_height = 1080 // 8
resolution_width = 1920 // 8

# load in the video data
def load_video(fname, patch_size=32, num_patches_per_video=1024):
    vr = VideoReader(fname, ctx=cpu(0))

    # randomly select frames with replacement
    frame_idx_list = np.random.choice(len(vr), num_patches_per_video, replace=True)

    patch_arr = np.zeros((num_patches_per_video, patch_size**2))
    # loop over each frame and select a random image patch and normalize it
    for i, idx in enumerate(frame_idx_list):
        frame = vr[idx].asnumpy()
        grayscale_frame = np.mean(frame, axis=-1)
        downscale_frame = cv2.resize(grayscale_frame, (resolution_width, resolution_height), interpolation=cv2.INTER_AREA)
        while True:
            x = np.random.randint(0, downscale_frame.shape[0] - patch_size)
            y = np.random.randint(0, downscale_frame.shape[1] - patch_size)
            patch = downscale_frame[x:x+patch_size, y:y+patch_size].flatten()
            patch_sd = np.std(patch)
            if patch_sd > 0:
                patch_arr[i, :] = (patch - np.mean(patch)) / patch_sd
                break

    return patch_arr

data_dir = 'PCA_then_ICA'
if not os.path.exists(data_dir):
    os.makedirs(data_dir)

# check if numpy array already exists
if os.path.exists(data_dir + "/image_patches.npy"):
    print("Image patches already exist. Loading...")
    image_patches = np.load(data_dir + "/image_patches.npy")
    print(f"Collected patches shape: {image_patches.shape}")
    print("Skipping to PCA")
else:
    environments = ["field", "forest", "orchard", "tall_grass", "pond"]
    patch_size = 32
    num_patches_per_video = 5000
    num_videos = int(2*len(environments)*10)

    all_fnames = []
    for environment_name in environments:
        for fname in glob.glob(vid_data_dir + environment_name + "/*.MP4"):
            if "free_moving" not in fname:
                all_fnames.append(fname)

    # parallelize the loading of the video data
    num_cores = min(multiprocessing.cpu_count(), len(all_fnames))
    print(f"Using {num_cores} cores to load video data")
    # load_video(fname, patch_size=patch_size, num_patches_per_video=num_patches_per_video)

    image_patches = Parallel(n_jobs=num_cores)(delayed(load_video)(fname, patch_size=patch_size, num_patches_per_video=num_patches_per_video) for fname in all_fnames)

    image_patches = np.vstack(image_patches)

    print(f"Collected patches shape: {image_patches.shape}")

    print("Saving image patches")
    np.save(data_dir + "/image_patches.npy", image_patches)

# Do ICA
print("Running ICA...")
ICA_model = FastICA(n_components=104, max_iter=10000, whiten='unit-variance')
ICA_model.fit(image_patches)

# save the ica results
np.save(data_dir + "/ica_components_FastICA.npy", ICA_model.components_)
np.save(data_dir + "/mixing_matrix_FastICA.npy", ICA_model.mixing_)
np.save(data_dir + "/whitening_matrix_FastICA.npy", ICA_model.whitening_)
np.save(data_dir + "/ica_mean_FastICA.npy", ICA_model.mean_)
# np.save(data_dir + "/ica_n_iter_FastICA_prewhiten.npy", ICA_model.n_iter_)

print("Processing complete. Outputs saved.")
