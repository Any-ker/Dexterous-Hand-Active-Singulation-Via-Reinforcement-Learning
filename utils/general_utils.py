import os
import pickle
import numpy as np
import torch
import os.path as osp
import yaml
import open3d as o3d
import trimesh

# These paths are used across the project, so keep them handy.
BASE_DIR = osp.dirname(osp.dirname(osp.dirname(osp.realpath(__file__))))
LOG_DIR = osp.join(osp.dirname(BASE_DIR), "Logs")
ASSET_DIR = osp.join(osp.dirname(BASE_DIR), "Assets")
print("================ Run ================")
print("BASE_DIR", BASE_DIR)
print("LOG_DIR:", LOG_DIR)
print("ASSET_DIR:", ASSET_DIR)
print("================ Run ================")


def load_yaml(path):
    with open(path, "r") as f:
        return yaml.safe_load(f)


def save_yaml(path, data):
    with open(path, "w") as f:
        yaml.safe_dump(data, f)


def save_pickle(path, data):
    with open(path, "wb") as f:
        pickle.dump(data, f)


def save_list_strings(path, lines):
    with open(path, "w") as f:
        for line in lines:
            f.write(f"{line}\n")


def compute_time_encoding(step_tensor, dim):
    """Simple sinusoidal encoding similar to the Transformer positional encoding."""
    step_tensor = step_tensor.to(torch.float32)
    enc = torch.zeros(step_tensor.shape[0], dim, device=step_tensor.device)
    positions = step_tensor.unsqueeze(1)
    div_term = torch.exp(
        torch.arange(0, dim, 2, dtype=torch.float32, device=step_tensor.device)
        * (-torch.log(torch.tensor(10000.0)) / dim)
    )
    enc[:, 0::2] = torch.sin(positions * div_term)
    enc[:, 1::2] = torch.cos(positions * div_term)
    return enc


def simplify_trimesh(mesh, ratio=0.1, min_faces=None):
    """Use Open3D to downsample a mesh and return it as a trimesh object."""
    simple = o3d.geometry.TriangleMesh()
    simple.vertices = o3d.utility.Vector3dVector(mesh.vertices)
    simple.triangles = o3d.utility.Vector3iVector(mesh.faces)
    target = int(len(simple.triangles) * ratio)
    if min_faces is not None:
        target = max(target, min_faces)
    simple = simple.simplify_quadric_decimation(target)
    vertices = np.asarray(simple.vertices)
    triangles = np.asarray(simple.triangles)
    return trimesh.Trimesh(vertices=vertices, faces=triangles, process=True)


@torch.jit.script
def batch_quat_apply(quat, vec):
    """Rotate batches of vectors by batches of quaternions."""
    shape = vec.shape
    quat = quat.unsqueeze(1)
    xyz = quat[:, :, :3]
    t = torch.cross(xyz, vec, dim=-1) * 2.0
    return (vec + quat[:, :, 3:] * t + torch.cross(xyz, t, dim=-1)).view(shape)


@torch.jit.script
def batch_sided_distance(src, dst):
    """Compute the distance from each src point to the nearest dst point."""
    dist = torch.cdist(src, dst)
    result, _ = torch.min(dist, dim=-1)
    return result


def compute_trajectory_valids(observations):
    """Compute validity mask for trajectories based on object positions.
    
    Args:
        observations: numpy array of shape (N, T, 3) containing object positions
        
    Returns:
        valids: numpy array of shape (N, T) indicating valid timesteps
    """
    # Check if object position is non-zero (indicating valid state)
    # Shape: (N, T, 3) -> (N, T)
    pos_norm = np.linalg.norm(observations, axis=-1)
    valids = (pos_norm > 1e-6).astype(np.float32)
    return valids


def set_seed(seed, torch_deterministic=False):
    """Utility function shared by multiple scripts."""
    if seed == -1:
        seed = 42 if torch_deterministic else np.random.randint(0, 10000)
    print(f"Setting seed: {seed}")
    np.random.seed(seed)
    torch.manual_seed(seed)
    torch.cuda.manual_seed_all(seed)
    os.environ["PYTHONHASHSEED"] = str(seed)
    if torch_deterministic:
        os.environ["CUBLAS_WORKSPACE_CONFIG"] = ":4096:8"
        torch.backends.cudnn.benchmark = False
        torch.backends.cudnn.deterministic = True
        torch.use_deterministic_algorithms(True)
    else:
        torch.backends.cudnn.benchmark = True
        torch.backends.cudnn.deterministic = False
    return seed
