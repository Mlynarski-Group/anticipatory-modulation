"""
This script simulates an agent moving in a spatially heterogeneous environment.
Author: Jonathan Gant
Date: 07.05.2025
"""

# imports
import numpy as np
import h5py
import os
from scipy.ndimage import gaussian_filter

class Environment:
    def __init__(self, x_min, x_max, sigma_sp, n_bins=50, scale=1, seed=0, zscore=False):
        self.x_min = x_min
        self.x_max = x_max
        self.n_bins = int(n_bins)
        self.sigma_sp = sigma_sp
        self.scale = scale
        self.seed = seed
        self.zscore = zscore
        np.random.seed(seed)
        self.stimuli, self.mask = self.gen_stimuli()

    def gen_stimuli(self):
        background = np.random.normal(size=(self.n_bins, self.n_bins))
        mask = self.gen_corr_mask()
        stimuli = background + mask
        return stimuli, mask

    def gen_corr_mask(self):
        initial_mask = np.random.normal(size=(self.n_bins, self.n_bins))
        mask = gaussian_filter(initial_mask, self.sigma_sp * self.n_bins / np.abs(self.x_max - self.x_min))
        if self.zscore:
            mask = (mask - np.mean(mask)) / np.std(mask)
        else:
            mask = (mask - np.min(mask)) / (np.max(mask) - np.min(mask)) * 2 - 1
            mask *= self.scale
        return mask

class Agent:
    def __init__(self, alpha=0.5, v_r=1, start_pos=np.array([0, 0]), sim_seed=0, sensor_vec=np.array([0, 0])):
        self.alpha = alpha
        self.v_r = v_r
        self.x = start_pos
        self.x_hist = np.atleast_2d(self.x)  # Initialize as a 2D NumPy array
        np.random.seed(sim_seed)
        self.velocity = self.gen_rand_velocity(v_r, sim_seed)
        self.velocity_hist = np.array([self.velocity])
        self.sensor_vec = sensor_vec
        self.stimuli_hist = []
        self.stimuli_hist2 = []

    def rot_mat(self, theta):
        return np.array([[np.cos(theta), -np.sin(theta)], [np.sin(theta), np.cos(theta)]])
    
    def gen_rand_velocity(self, v_r, sim_seed):
        velocity = v_r * np.array([1, 0])
        theta = np.random.uniform(0, 2 * np.pi)
        return self.rot_mat(theta) @ velocity.T

    def update_velocity(self, new_velocity):
        self.velocity = new_velocity
        self.velocity_hist = np.append(self.velocity_hist, np.atleast_2d(new_velocity), axis=0)

    def update_position(self, new_x):
        self.x = new_x
        self.x_hist = np.append(self.x_hist, np.atleast_2d(new_x), axis=0)

    def update_stimuli(self, current_stimulus, current_stimulus2):
        self.stimuli_hist.append(current_stimulus)
        self.stimuli_hist2.append(current_stimulus2)

if __name__ == "__main__":
    # Simulation parameters
    x_min, x_max = -50, 50
    sigma_sp_values = [0.1, 0.2, 0.5, 1, 2, 5]  # Different sigma_sp values
    scale = 100
    alpha = 0.5
    velocities = [0.1, 0.2, 1, 2, 5]  # Velocities to consider
    n_bins = 100
    T = 50
    dt = 1/30
    times = np.arange(0, T, dt)
    sensor_radii = [10]  # Radii for sensor sampling points
    num_sensors = 8  # Number of sensors on the circle

    noise_level = 0.001 # 0.1

    num_envs = 10  # Number of different environments
    num_sims = 10 # Number of simulations per environment

    x_mesh = np.linspace(x_min, x_max, n_bins)

    # make results directory if it doesn't exist
    os.makedirs("../results", exist_ok=True)

    # Save all data to a single HDF5 file
    with h5py.File("../results/simulation_results_full_low_noise_more_env_more_sim_more_v_more_sigma_redo_zscore.hdf5", "w") as f:
        for sigma_sp in sigma_sp_values:
            grp_sigma = f.create_group(f"sigma_sp_{sigma_sp}")
            for env_idx in range(num_envs):
                env_seed = env_idx
                env = Environment(x_min, x_max, sigma_sp, n_bins=n_bins, scale=scale, seed=env_seed, zscore=True)
                grp_env = grp_sigma.create_group(f"environment_{env_idx}")
                grp_env.create_dataset("mask", data=env.mask)
                grp_env.attrs["params"] = [x_min, x_max, n_bins, sigma_sp, scale, env_seed]

                for v_r in velocities:
                    grp_velocity = grp_env.create_group(f"velocity_{v_r}")
                    for radius in sensor_radii:
                        grp_radius = grp_velocity.create_group(f"radius_{radius}")
                        sensor_vecs = [
                            np.array([radius * np.cos(theta), radius * np.sin(theta)])
                            for theta in np.linspace(0, 2 * np.pi, num_sensors, endpoint=False)
                        ]

                        for sim_idx in range(num_sims):
                            sim_seed = sim_idx
                            start_pos = np.random.uniform(x_min, x_max, 2)
                            agent = Agent(alpha, v_r, start_pos, sim_seed=sim_seed)
                            # Simulation loop
                            for t in range(len(times)):
                                background = np.random.normal(scale=noise_level, size=(env.n_bins, env.n_bins))
                                stimuli = background + env.mask
                                noise = agent.v_r * np.random.normal(size=2)
                                velocity = agent.velocity
                                new_x = agent.x + (agent.alpha * velocity + (1 - agent.alpha) * noise)
                                velocity = new_x - agent.x

                                # Reflective boundary conditions
                                if new_x[0] < x_min or new_x[0] > x_max:
                                    new_x[0] = max(min(new_x[0], x_max), x_min)
                                    velocity[0] = -velocity[0]
                                if new_x[1] < x_min or new_x[1] > x_max:
                                    new_x[1] = max(min(new_x[1], x_max), x_min)
                                    velocity[1] = -velocity[1]

                                agent.update_velocity(velocity)
                                agent.update_position(new_x)

                                x1_idx = np.digitize(agent.x[0], x_mesh) - 1
                                x2_idx = np.digitize(agent.x[1], x_mesh) - 1
                                current_stimulus = stimuli[x1_idx, x2_idx]

                                # Sample stimuli at sensor points
                                sensor_stimuli = []
                                for sensor_vec in sensor_vecs:
                                    x1_idx_s = np.digitize(agent.x[0] + sensor_vec[0], x_mesh) - 1
                                    x2_idx_s = np.digitize(agent.x[1] + sensor_vec[1], x_mesh) - 1
                                    sensor_stimuli.append(stimuli[x1_idx_s, x2_idx_s])

                                agent.update_stimuli(current_stimulus, sensor_stimuli)

                            # Save agent data
                            grp_agent = grp_radius.create_group(f"simulation_{sim_idx}")
                            grp_agent.create_dataset("trajectory", data=agent.x_hist)
                            grp_agent.create_dataset("stimulus_hist", data=agent.stimuli_hist)
                            grp_agent.create_dataset("sensor_stimuli", data=np.array(agent.stimuli_hist2))
                            grp_agent.attrs["params"] = [alpha, v_r, sim_seed, radius]

        # Save simulation parameters
        grp_sim = f.create_group("simulation_params")
        grp_sim.attrs["params"] = [dt, T, num_envs, num_sims, num_sensors]
        grp_sim.attrs["sensor_radii"] = sensor_radii
        grp_sim.attrs["sigma_sp_values"] = sigma_sp_values