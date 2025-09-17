"""
This notebook loads in videos of natural scenes collected via the video collection protocol, 
and computes the response of a set of Gabor filters.

Author: Jonathan Gant
Date: 04.08.2023
"""

# import statements
import numpy as np
from tqdm import tqdm
import cv2
import glob
import os
import h5py
from decord import VideoReader
from decord import cpu
import sys

data_group = 'nat_videos'

# define parameters of the video
data_dir = "../data/" + data_group + "/"

resolution_height = 1080
resolution_width = 1920

# load in the video data
def load_video(fname):
    print("attempting to load the video " + fname)
    try:
        vr = VideoReader(fname, ctx=cpu(0))
        print("successfully found the video")
        print('video frames:', len(vr))
    except:
        print("failed to find the video :(")
        return None

    video = np.empty((len(vr), resolution_height, resolution_width), dtype=np.float32)
    for i in range(len(vr)):
        frame = vr[i].asnumpy()
        grayscale_frame = np.mean(frame, axis=-1)
        video[i] = grayscale_frame

    # normalize the video
    video -= np.mean(video)
    video /= np.std(video)

    return video
    

# define the Gabor filter function
def gabor_filter_func(
    sigma, theta, gamma, wavelength, phase, filt_size, x_offset=0, y_offset=0
):
    y, x = np.meshgrid(
        np.arange(filt_size[1]) - filt_size[1] // 2,
        np.arange(filt_size[0]) - filt_size[0] // 2,
    )
    x = x - x_offset
    y = y + y_offset
    theta_rad = np.pi * theta / 180
    x_prime = x * np.cos(theta_rad) + y * np.sin(theta_rad)
    y_prime = -x * np.sin(theta_rad) + y * np.cos(theta_rad)
    filter = np.exp(-0.5 * (x_prime**2 + (gamma * y_prime) ** 2) / sigma**2) * np.cos(
        2 * np.pi * x_prime / wavelength + np.pi * phase / 180)
    filter /= np.linalg.norm(filter)
    return filter


# create a bank of Gabor filters with different orientations, phases, and spatial frequencies
orientation_arr = np.linspace(0, 157.5, 8)
phase_arr = np.linspace(0, 270, 4)
position_arr = np.array([[-resolution_height//4, resolution_width//4], [-resolution_height//4, 0], [-resolution_height//4, -resolution_width//4], [0, resolution_width//4], [0, 0], [0, -resolution_width//4], [resolution_height//4, resolution_width//4], [resolution_height//4, 0], [resolution_height//4, -resolution_width//4]])

pixels_per_degree = 20

freq_arr = np.array([0.03, 0.04, 0.05, 0.06, 0.07, 0.08, 0.09, 0.1, 0.11, 0.12])
wavelength_arr = pixels_per_degree/freq_arr
print(wavelength_arr)

filter_size = (resolution_height, resolution_width)

all_gabor_filters = []

for l, wavelength in enumerate(wavelength_arr):
    filt_size = int(1.5*wavelength)
    # check if the filter size is odd
    if filt_size % 2 == 1:
        filt_size += 1
    gabor_filter_bank = np.zeros((len(orientation_arr), len(phase_arr), filt_size, filt_size))
    for i, orientation in tqdm(enumerate(orientation_arr)):
        print("orientation: " + str(orientation))
        for j, phase in enumerate(phase_arr):
            gabor_filter = gabor_filter_func(
                sigma=wavelength/3,
                theta=orientation,
                wavelength=wavelength,
                phase=phase,
                gamma=1,
                filt_size=(filt_size, filt_size),
                x_offset=0,
                y_offset=0,
            )
            gabor_filter_bank[i, j] = gabor_filter
    all_gabor_filters.append(gabor_filter_bank)

def gen_gabor_response(scene, condition, all_gabor_responses):
    if condition == '':
        group = all_gabor_responses.create_group(scene.replace("/", ""))
    else:
        group = all_gabor_responses.create_group(scene.replace("/", "") + '_' + condition.replace("/", ""))
    for fname in tqdm(glob.glob(data_dir + scene + condition + "*.MP4")):
        if 'moving' not in fname:
            video = load_video(fname)
            if video is None:
                continue

            gabor_responses = np.zeros((len(orientation_arr), len(phase_arr), len(wavelength_arr), len(position_arr), video.shape[0]))

            # compute the Gabor filter response
            for l, wavelength in enumerate(wavelength_arr):
                filters_per_wavelength = all_gabor_filters[l]
                filt_size = filters_per_wavelength.shape[2]
                for m, position in enumerate(position_arr):
                    x = position[1]
                    y = position[0]
                    for i in range(video.shape[0]):
                        if i % 60 == 0:
                            x += int(np.random.choice([-1, 1]) * np.max(np.random.normal(200, 60), 0))
                            y += int(np.random.choice([-1, 1]) * np.max(np.random.normal(20, 6), 0))
                            # check if the position is within the bounds of the video
                            x = np.clip(x, filt_size//2, resolution_width - filt_size//2)
                            y = np.clip(y, filt_size//2, resolution_height - filt_size//2)
                        gabor_responses[:, :, l, m, i] = np.tensordot(filters_per_wavelength, video[i, y-filt_size//2:y+filt_size//2, x-filt_size//2:x+filt_size//2], axes=([2, 3], [0, 1]))

            key_fname = os.path.basename(fname)[:-4] # remove the '.mp4 and prefix

            group.create_dataset(key_fname, data=gabor_responses)

# make results directory if it doesn't exist
os.makedirs('../results/', exist_ok=True)

# for each video compute the Gabor filter response
all_gabor_responses = h5py.File('../results/' + data_group + '_gabor_responses_full_res_more_low_freq_z_score_eye_movements_2s_interval_stat_only.h5', 'w')

gen_gabor_response('field/', '', all_gabor_responses)
gen_gabor_response('forest/', '', all_gabor_responses)
gen_gabor_response('orchard/', '', all_gabor_responses)
gen_gabor_response('tall_grass/', '', all_gabor_responses)
gen_gabor_response('pond/', '', all_gabor_responses)

# save the orientation, phase, position, rf_size, and wavelength arrays
all_gabor_responses.create_dataset('orientation_arr', data=orientation_arr)
all_gabor_responses.create_dataset('phase_arr', data=phase_arr)
all_gabor_responses.create_dataset('position_arr', data=position_arr)
all_gabor_responses.create_dataset('wavelength_arr', data=wavelength_arr)
all_gabor_responses.create_dataset('freq_arr', data=freq_arr)

all_gabor_responses.close()