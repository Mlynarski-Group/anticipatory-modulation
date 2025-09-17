import numpy as np
import torch
from tqdm import tqdm
from joblib import Parallel, delayed
from torch.optim import Adam

data_dir = 'PCA_then_ICA' # 1024 patches per video, 20000 iterations

stationary_responses_nonrandom = np.load(data_dir + "/stationary_responses_nonrandom_multi_dist_FastICA.npy")
moving_responses_nonrandom = np.load(data_dir + "/moving_responses_nonrandom_multi_dist_FastICA.npy")

# now let's do this for a number of filters for a certain number of randomly drawn windows
num_windows = 1000
num_filters = 100
num_neighbors = 100
window_size = 150  # Assuming window_size is defined somewhere
num_epochs = 500  # Reduced the number of epochs for faster optimization
learning_rate = 0.005
reg_lmbd = 0.1

# define the loss function as the mean pearson correlation (use torch.corrcoef) between the residuals and the neighbors
def loss_fn(neighbors, ref, weights, reg_lmbd=0.01):
    residuals = ref - torch.matmul(neighbors, weights)
    
    # Stack the residuals and neighbors
    stacked = torch.cat([residuals, neighbors], dim=1).T

    # compute the correlation
    correlation_matrix = torch.corrcoef(stacked)
    
    # Extract the correlations with the new reference response
    correlations = torch.abs(correlation_matrix[0, 1:])
    
    # Compute the mean correlation
    mean_correlation = correlations.mean()

    # compute the L1 norm of the weights
    # weight_norm = torch.norm(weights, p=1)
    # compute the L2 norm of the weights
    weight_norm = torch.norm(weights, p=2)

    return mean_correlation + reg_lmbd*weight_norm

def process_filter_window(filt_idx, j):
    start_idx = np.random.randint(0, stationary_responses_nonrandom.shape[2] - window_size)
    end_idx = start_idx + window_size

    neighbors_moving = np.delete(moving_responses_nonrandom[:num_neighbors+1, 0, start_idx:end_idx], filt_idx, axis=0)
    neighbors_stationary = np.delete(stationary_responses_nonrandom[:num_neighbors+1, 0, start_idx:end_idx], filt_idx, axis=0)
    ref_moving = moving_responses_nonrandom[filt_idx, 0, start_idx:end_idx]
    ref_stationary = stationary_responses_nonrandom[filt_idx, 0, start_idx:end_idx]

    # subtract the mean
    neighbors_moving = neighbors_moving - np.mean(neighbors_moving, axis=1)[:, np.newaxis]
    neighbors_stationary = neighbors_stationary - np.mean(neighbors_stationary, axis=1)[:, np.newaxis]
    ref_moving = ref_moving - np.mean(ref_moving)
    ref_stationary = ref_stationary - np.mean(ref_stationary)

    # apply relu
    neighbors_moving = np.maximum(neighbors_moving, 0)
    neighbors_stationary = np.maximum(neighbors_stationary, 0)
    ref_moving = np.maximum(ref_moving, 0)
    ref_stationary = np.maximum(ref_stationary, 0)

    # Compute the correlation between the ref and neighbors
    correlation_moving = np.array([np.corrcoef(neighbors_moving[k, :], ref_moving)[0, 1] for k in range(num_neighbors)])
    correlation_stationary = np.array([np.corrcoef(neighbors_stationary[k, :], ref_stationary)[0, 1] for k in range(num_neighbors)])

    # optimize weights using PyTorch and gradient based methods
    # convert the data to torch tensors
    neighbors_moving = torch.tensor(neighbors_moving, dtype=torch.float32).T
    neighbors_stationary = torch.tensor(neighbors_stationary, dtype=torch.float32).T
    ref_moving = torch.tensor(ref_moving, dtype=torch.float32).view(-1, 1)
    ref_stationary = torch.tensor(ref_stationary, dtype=torch.float32).view(-1, 1)

    # guess randomly the weights for the moving and stationary case between 0 and 1
    moving_weights = torch.rand(neighbors_moving.shape[1], 1, dtype=torch.float32, requires_grad=True)
    stationary_weights = torch.rand(neighbors_stationary.shape[1], 1, dtype=torch.float32, requires_grad=True)

    # define the optimizer
    optimizer_moving = Adam([moving_weights], lr=learning_rate)
    optimizer_stationary = Adam([stationary_weights], lr=learning_rate)

    loss_history_moving = []
    weight_history_moving = []
    loss_history_stationary = []
    weight_history_stationary = []

    # run the optimization
    for i in range(num_epochs):
        # zero grad from previous iteration
        optimizer_moving.zero_grad()
        optimizer_stationary.zero_grad()
        # store the weights
        weight_history_moving.append(moving_weights.clone().detach().cpu().numpy())
        weight_history_stationary.append(stationary_weights.clone().detach().cpu().numpy())
        # compute and store the loss
        loss_moving = loss_fn(neighbors_moving, ref_moving, moving_weights, reg_lmbd=reg_lmbd)
        loss_history_moving.append(loss_moving.item())
        loss_stationary = loss_fn(neighbors_stationary, ref_stationary, stationary_weights, reg_lmbd=reg_lmbd)
        loss_history_stationary.append(loss_stationary.item())
        if i < num_epochs - 1:
            # backpropagate
            loss_moving.backward()
            loss_stationary.backward()
            # update weights
            optimizer_moving.step()
            optimizer_stationary.step()
            # Enforce positive weights
            with torch.no_grad():
                moving_weights.data = torch.relu(moving_weights.data)
                stationary_weights.data = torch.relu(stationary_weights.data)

    # cast the history to numpy arrays
    weight_history_moving = np.array(weight_history_moving)
    weight_history_stationary = np.array(weight_history_stationary)
    loss_history_moving = np.array(loss_history_moving)
    loss_history_stationary = np.array(loss_history_stationary)

    # Compute the residuals
    residuals_moving = ref_moving - torch.matmul(neighbors_moving, moving_weights)
    residuals_stationary = ref_stationary - torch.matmul(neighbors_stationary, stationary_weights)

    # Convert residuals to numpy arrays
    residuals_moving = residuals_moving.detach().cpu().numpy()
    residuals_stationary = residuals_stationary.detach().cpu().numpy()

    # Compute the correlation between the residuals and the neighbors
    correlation_moving_after = np.array([np.corrcoef(neighbors_moving[:, k].detach().cpu().numpy(), residuals_moving[:, 0])[0, 1] for k in range(num_neighbors)])
    correlation_stationary_after = np.array([np.corrcoef(neighbors_stationary[:, k].detach().cpu().numpy(), residuals_stationary[:, 0])[0, 1] for k in range(num_neighbors)])

    return (weight_history_moving, weight_history_stationary, loss_history_moving, loss_history_stationary, correlation_moving, correlation_stationary, correlation_moving_after, correlation_stationary_after, residuals_moving, residuals_stationary, start_idx)

results = Parallel(n_jobs=-1)(delayed(process_filter_window)(i, j) for i in tqdm(range(num_filters)) for j in range(num_windows))

# make the arrays and unpack the results
weights_moving = np.zeros((num_filters, num_windows, num_epochs, num_neighbors, 1))
weights_stationary = np.zeros((num_filters, num_windows, num_epochs, num_neighbors, 1))
losses_moving = np.zeros((num_filters, num_windows, num_epochs))
losses_stationary = np.zeros((num_filters, num_windows, num_epochs))
correlation_moving = np.zeros((num_filters, num_windows, num_neighbors))
correlation_stationary = np.zeros((num_filters, num_windows, num_neighbors))
correlation_moving_after = np.zeros((num_filters, num_windows, num_neighbors))
correlation_stationary_after = np.zeros((num_filters, num_windows, num_neighbors))
residuals_moving_arr = np.zeros((num_filters, num_windows, window_size, 1))
residuals_stationary_arr = np.zeros((num_filters, num_windows, window_size, 1))
start_indices = np.zeros((num_filters, num_windows))

for i in range(num_filters):
    for j in range(num_windows):
        (weights_moving[i, j, :, :, :], weights_stationary[i, j, :, :, :], losses_moving[i, j, :], losses_stationary[i, j, :],
         correlation_moving[i, j, :], correlation_stationary[i, j, :], correlation_moving_after[i, j, :], correlation_stationary_after[i, j, :],
         residuals_moving_arr[i, j, :, :], residuals_stationary_arr[i, j, :, :], start_indices[i, j]) = results[i * num_windows + j]
        
# save everything in a directory called decorrelation_neighbors_pytorch
data_dir = f'decorrelation_neighbors_FastICA_demean_relu_pytorch_l2_learning_rate_{learning_rate}_reg_lmbd_{reg_lmbd}_num_windows_{num_windows}_window_size_{window_size}'
# make the directory if it does not exist
import os
if not os.path.exists(data_dir):
    os.makedirs(data_dir)
np.save(data_dir + '/weights_moving', weights_moving)
np.save(data_dir + '/weights_stationary', weights_stationary)
np.save(data_dir + '/losses_moving', losses_moving)
np.save(data_dir + '/losses_stationary', losses_stationary)
np.save(data_dir + '/correlation_moving', correlation_moving)
np.save(data_dir + '/correlation_stationary', correlation_stationary)
np.save(data_dir + '/correlation_moving_after', correlation_moving_after)
np.save(data_dir + '/correlation_stationary_after', correlation_stationary_after)
np.save(data_dir + '/residuals_moving', residuals_moving_arr)
np.save(data_dir + '/residuals_stationary', residuals_stationary_arr)
np.save(data_dir + '/start_indices', start_indices)
