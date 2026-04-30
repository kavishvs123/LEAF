import torch
import numpy as np
from cfg import args


def move_data(data, device):
    for key in data.keys():
        if isinstance(data[key], torch.Tensor):
            data[key] = data[key].to(device)
    return data

def time_idx_to_datetime(time_idx, offset=0):
    time = args.start_time + args.time_delta * (time_idx + offset)
    return time

def RMSE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        rmse = np.square(np.abs(y_pred - y_true))
        rmse = np.nan_to_num(rmse * mask)
        rmse = np.sqrt(np.mean(rmse))
        return rmse


def MAE(y_true, y_pred):
    with np.errstate(divide="ignore", invalid="ignore"):
        mask = np.not_equal(y_true, 0)
        mask = mask.astype(np.float32)
        mask /= np.mean(mask)
        mae = np.abs(y_pred - y_true)
        mae = np.nan_to_num(mae * mask)
        mae = np.mean(mae)
        return mae


def MAPE(y_true, y_pred, null_val=0):
    with np.errstate(divide="ignore", invalid="ignore"):
        if np.isnan(null_val):
            mask = ~np.isnan(y_true)
        else:
            mask = np.not_equal(y_true, null_val)
        mask = mask.astype("float32")
        mask /= np.mean(mask)
        mape = np.abs(np.divide((y_pred - y_true).astype("float32"), y_true))
        mape = np.nan_to_num(mask * mape, nan=0)
        return np.mean(mape) * 100


def metric(y_pred, y_true, verbose=False, time_dim=1):
    if isinstance(y_pred, torch.Tensor):
        y_pred = y_pred.detach().cpu().numpy()
    if isinstance(y_true, torch.Tensor):
        y_true = y_true.detach().cpu().numpy()
    mae = MAE(y_true, y_pred)
    rmse = RMSE(y_true, y_pred)
    mape = MAPE(y_true, y_pred)
    if verbose:
        mae_list, rmse_list, mape_list = [], [], []
        for idx in range(y_pred.shape[time_dim]):
            y_pred_slice = np.take(y_pred, indices=idx, axis=time_dim)
            y_true_slice = np.take(y_true, indices=idx, axis=time_dim)
            mae_list.append(MAE(y_true_slice, y_pred_slice))
            rmse_list.append(RMSE(y_true_slice, y_pred_slice))
            mape_list.append(MAPE(y_true_slice, y_pred_slice))
        return mae, rmse, mape, mae_list, rmse_list, mape_list
    return mae, rmse, mape


def _normalize(x, mean_value, std_value):
    if isinstance(x, torch.Tensor):
        mean = torch.tensor(mean_value, device=x.device)
        std = torch.tensor(std_value, device=x.device)
        while len(mean.shape) < len(x.shape):
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return (x - mean) / std
    elif isinstance(x, np.ndarray):
        mean = mean_value
        std = std_value
        while len(mean.shape) < len(x.shape):
            mean = mean[np.newaxis, ...]
            std = std[np.newaxis, ...]
        return (x - mean) / std
    else:
        raise ValueError('x should be torch.Tensor or np.ndarray')


def normalize_feat(feat):
    return _normalize(feat, args.feature_mean, args.feature_std)

def normalize_output(output):
    return _normalize(output, args.output_mean, args.output_std)

def _denormalize(x, mean_value, std_value):
    if isinstance(x, torch.Tensor):
        mean = torch.tensor(mean_value, device=x.device)
        std = torch.tensor(std_value, device=x.device)
        while len(mean.shape) < len(x.shape):
            mean = mean.unsqueeze(0)
            std = std.unsqueeze(0)
        return x * std + mean
    elif isinstance(x, np.ndarray):
        mean = mean_value
        std = std_value
        while len(mean.shape) < len(x.shape):
            mean = mean[np.newaxis, ...]
            std = std[np.newaxis, ...]
        return x * std + mean
    else:
        raise ValueError('x should be torch.Tensor or np.ndarray')
    
def denormalize_feat(feat):
    return _denormalize(feat, args.feature_mean, args.feature_std)

def denormalize_output(output):
    return _denormalize(output, args.output_mean, args.output_std)

#--- Beginning Change ---#

def get_float_adj(distance_df, num_nodes, node_order_path):
    # 1. Create the same mapping used in get_dataloader
    idx2nodeid = {}
    if node_order_path is not None:
        # This part handles PEMS03 specific ordering
        with open(node_order_path, 'r') as f:
            for i, line in enumerate(f):
                idx2nodeid[int(line.strip())] = i
    else:
        # This handles PEMS04 and PEMS08 (direct mapping)
        idx2nodeid = {i: i for i in range(num_nodes)}

    # 2. Initialize the float adjacency matrix
    adj_float = np.zeros((num_nodes, num_nodes), dtype=np.float32)

    # 3. Fill with actual distance values (the 'd' or third column)
    # distance_df columns are usually: [from, to, cost/distance]
    for row in distance_df.values:
        u_id, v_id, dist = int(row[0]), int(row[1]), float(row[2])
        
        if u_id in idx2nodeid and v_id in idx2nodeid:
            u, v = idx2nodeid[u_id], idx2nodeid[v_id]
            adj_float[u, v] = dist
            adj_float[v, u] = dist # Assumes undirected graph

    return adj_float

def norm_adj(adj):
    """
    Symmetrically normalizes the adjacency matrix: D^-1/2 * A * D^-1/2
    Works for both single matrices and batches of matrices.
    """
    # adj shape is usually (Batch, N, N) or (1, N, N)
    
    # 1. Calculate the degree matrix (sum of each row)
    # Add a tiny epsilon (1e-9) to prevent division by zero for isolated nodes
    deg = torch.sum(adj, dim=-1)
    
    # 2. Compute D^-1/2
    deg_inv_sqrt = torch.pow(deg + 1e-9, -0.5)
    
    # 3. Handle potential infinite values if any degrees were exactly 0
    deg_inv_sqrt[torch.isinf(deg_inv_sqrt)] = 0.
    
    # 4. Create a diagonal matrix from the degree vector
    d_mat_inv_sqrt = torch.diag_embed(deg_inv_sqrt)
    
    # 5. Return the normalized matrix: D^-1/2 @ A @ D^-1/2
    return d_mat_inv_sqrt @ adj @ d_mat_inv_sqrt
#--- End Change ---#
