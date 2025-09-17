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
from decord import cpu, gpu
from utilities import (
    calc_group_variance,
    calc_group_mean,
)
import sys

data_group = 'nat_videos'

# define parameters of the video
data_dir = "../data/" + data_group + "/"

resolution_height = 1080
resolution_width = 1920

# load in the video data
def load_video(fname):
    video = []

    print("attempting to load the video " + fname)
    # create separate thread to load in the video frames
    try:
        vr = VideoReader(fname, ctx=cpu(0))
        print("successfully found the video")
        print('video frames:', len(vr))
    except:
        print("failed to find the video :(")

    for i in range(len(vr)):
        frame = vr[i].asnumpy()
        grayscale_frame = np.mean(frame, axis=-1)
        # downscale_frame = cv2.resize(grayscale_frame, (resolution_width, resolution_height), interpolation=cv2.INTER_AREA)
        video.append(grayscale_frame)

    # define as an array
    video = np.array(video)
    
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
    x_prime = x * np.cos(np.pi * theta / 180) + y * np.sin(np.pi * theta / 180)
    y_prime = -x * np.sin(np.pi * theta / 180) + y * np.cos(np.pi * theta / 180)
    filter = np.exp(-0.5 * (x_prime**2 + (gamma * y_prime) ** 2) / sigma**2) * np.cos(
        2 * np.pi * x_prime / wavelength + np.pi * phase / 180)
    filter /= np.linalg.norm(filter)
    return filter


# create a bank of Gabor filters with different orientations, phases, and spatial frequencies
orientation_arr = np.linspace(0, 157.5, 8)
phase_arr = np.linspace(0, 270, 4)
position_arr = np.array([[-resolution_height//4, resolution_width//4], [-resolution_height//4, 0], [-resolution_height//4, -resolution_width//4],
                         [0, resolution_width//4], [0, 0], [0, -resolution_width//4],
                         [resolution_height//4, resolution_width//4], [resolution_height//4, 0], [resolution_height//4, -resolution_width//4]])

pixels_per_degree = 20

low_freq_arr = np.linspace(0.02, 0.36, 35)
high_freq_arr = np.arange(2, 6.12, .12)
freq_arr = np.concatenate((low_freq_arr, high_freq_arr))
wavelength_arr = pixels_per_degree/freq_arr
print(wavelength_arr)

sigma_factor = 3

filter_size = (resolution_height, resolution_width)

gabor_filter_bank = np.zeros((len(orientation_arr), len(phase_arr), len(wavelength_arr), len(position_arr), resolution_height, resolution_width))

for i, orientation in tqdm(enumerate(orientation_arr)):
    print("orientation: " + str(orientation))
    for j, phase in enumerate(phase_arr):
        for l, wavelength in enumerate(wavelength_arr):
            for m, position in enumerate(position_arr):
                gabor_filter = gabor_filter_func(
                    sigma=wavelength/sigma_factor, # usually divide by 3
                    theta=orientation,
                    wavelength=wavelength,
                    phase=phase,
                    gamma=1,
                    filt_size=filter_size,
                    x_offset=position[0],
                    y_offset=position[1],
                )
                gabor_filter_bank[i, j, l, m, :, :] = gabor_filter 

def gen_gabor_response(scene, condition, all_gabor_responses):
    if condition == '':
        group = all_gabor_responses.create_group(scene.replace("/", ""))
    else:
        group = all_gabor_responses.create_group(scene.replace("/", "") + '_' + condition.replace("/", ""))
    for fname in tqdm(glob.glob(data_dir + scene + condition + "*.MP4")):    
        video = load_video(fname)

        gabor_responses = np.tensordot(gabor_filter_bank, video, axes=([-2, -1], [1, 2]))
        print(np.shape(gabor_responses))
    
        key_fname = os.path.basename(fname)[:-4] # remove the '.mp4 and prefix

        group.create_dataset(key_fname, data=gabor_responses)

# make results directory if it doesn't exist
os.makedirs('../results/', exist_ok=True)

# for each video compute the Gabor filter response
all_gabor_responses = h5py.File('../results/' + data_group + '_gabor_responses_full_res_z_score_more_freq.h5', 'w')
# all_gabor_responses = h5py.File('../results/' + data_group + '_gabor_responses_full_res_z_score_half_size.h5', 'w')
# all_gabor_responses = h5py.File('../results/' + data_group + '_gabor_responses_full_res_z_score_full_size.h5', 'w')

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
# save the sigma factor
all_gabor_responses.create_dataset('sigma_factor', data=sigma_factor)

all_gabor_responses.close()
