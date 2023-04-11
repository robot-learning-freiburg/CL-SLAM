import copy
import pickle
from typing import Dict, List, Optional, Tuple, Union

import cv2
import matplotlib.pyplot as plt
import numpy as np
from torch import Tensor

from depth_pose_prediction.networks.layers import BackprojectDepth
from slam.meshlab import MeshlabInf


def save_data(fname, obj):
    with open(fname, 'wb') as f:
        pickle.dump(obj, f)


def load_data(fname):
    with open(fname, 'rb') as f:
        data = pickle.load(f)
    return data


def depth_to_pcl(backproject_depth: BackprojectDepth,
                 inv_camera_matrix: Tensor,
                 depth: Tensor,
                 image: Tensor,
                 batch_size: int = 1,
                 dist_threshold: float = np.inf):
    pcl = backproject_depth(depth, inv_camera_matrix)
    pcl = pcl.squeeze().cpu().detach().numpy()[:3, :].T
    color = image.view(batch_size, 3, -1).squeeze().cpu().detach().numpy().T
    pcl = np.c_[pcl, color]
    if not np.isinf(dist_threshold):
        dist = np.linalg.norm(pcl[:, :3], axis=1)
        pcl = pcl[dist < dist_threshold, :]
    return pcl


def pcl_to_image(
    pcl: np.ndarray,
    camera_matrix: np.ndarray,
    image_shape: Tuple[int, int],
) -> np.ndarray:
    projection = cv2.projectPoints(pcl[:, :3].astype(np.float64), (0, 0, 0), (0, 0, 0),
                                   camera_matrix, np.zeros(4))[0].squeeze()
    image = np.zeros((image_shape[0], image_shape[1], 3))
    depth = np.ones((image_shape[0], image_shape[1], 1)) * np.inf
    for i, (u, v) in enumerate(projection):
        u, v = int(np.floor(u)), int(np.floor(v))
        if not (0 <= v < image_shape[0] and 0 <= u < image_shape[1]):
            continue
        distance = np.linalg.norm(pcl[i, :3])
        if distance < depth[v, u]:
            depth[v, u] = distance
            image[v, u] = pcl[i, 3:]
    return image


def save_point_cloud(
    filename: str,
    pcl: Union[np.ndarray, List[np.ndarray]],
    global_pose_list: Optional[np.ndarray] = None,
    verbose: bool = True,
) -> None:
    if global_pose_list is not None:
        accumulated_pcl = accumulate_pcl(pcl, global_pose_list)
    else:
        accumulated_pcl = pcl
    meshlab = MeshlabInf()
    meshlab.add_points(accumulated_pcl)
    meshlab.write(filename, verbose=verbose)


def accumulate_pcl(pcl_list: List[np.ndarray], global_pose_list: np.ndarray) -> np.ndarray:
    accumulated_pcl = []
    for i, (pcl, tmat) in enumerate(zip(pcl_list, global_pose_list)):
        accumulated_pcl.append(np.c_[(np.c_[pcl[:, :3], np.ones(
            (pcl.shape[0], 1))] @ tmat.T)[:, :3], pcl[:, 3:]])
    accumulated_pcl = np.concatenate(accumulated_pcl)
    return accumulated_pcl


def generate_figure(
    batch_i,
    image,
    depth,
    image_pcl,
    gt_xz,
    pred_xz,
    save_figure: bool = True,
) -> None:
    fig = plt.figure(figsize=(8, 10))
    plt.subplot(411)
    plt.imshow(image)
    plt.axis('off')
    plt.title('Current frame')
    plt.subplot(412)
    vmax = np.percentile(depth, 95)
    plt.imshow(depth, cmap='magma_r', vmax=vmax)
    plt.axis('off')
    plt.title(f'Predicted depth (vmax={vmax:.3f})')
    plt.subplot(413)
    plt.imshow(image_pcl)
    plt.axis('off')
    plt.title('Projected PCL (w/o current frame)')
    plt.subplot(414)
    plt.plot(gt_xz[:, 0], gt_xz[:, 1], label='gt')
    plt.plot(pred_xz[:, 0], pred_xz[:, 1], label='pred')
    plt.xlim((-400, 15))
    plt.ylim((-60, 160))
    # plt.xlim((-10, 10))
    # plt.ylim((-10, 10))
    plt.legend()
    plt.tight_layout()
    if save_figure:
        plt.savefig(f'./figures/sequence_08/{batch_i:05}.png', )
    else:
        plt.show()
    plt.close(fig)


# =============================================================================
# Adapted from:
# https://github.com/Huangying-Zhan/kitti-odom-eval/blob/master/kitti_odometry.py


def scale_optimization(pred_poses, gt_poses):
    """ Optimize scaling factor
    """

    # 2D trajectory
    if isinstance(pred_poses, np.ndarray) and pred_poses.shape[1] == 2:
        scaling = scale_lse_solver(pred_poses, gt_poses)
        pred_scaled = scaling * pred_poses
    # 3D poses (6 DoF)
    elif isinstance(pred_poses, list) and pred_poses[0].shape == (4, 4):
        # Scale only translation but keep rotation
        pred_xyz = np.asarray([p[:3, 3] for p in pred_poses])
        gt_xyz = np.asarray([p[:3, 3] for p in gt_poses])
        scaling = scale_lse_solver(pred_xyz, gt_xyz)
        pred_scaled = copy.deepcopy(pred_poses)
        for p in pred_scaled:
            p[:3, 3] *= scaling
    else:
        assert False
    return pred_scaled, scaling


def scale_lse_solver(X, Y):
    """Least-squares-error solver
    Compute optimal scaling factor so that s(X)-Y is minimum
    Args:
        X (KxN array): current data
        Y (KxN array): reference data
    Returns:
        scale (float): scaling factor
    """
    scale = np.sum(X * Y) / np.sum(X**2)
    return scale


def trajectory_distances(poses):
    """Compute distance for each pose w.r.t frame-0
    """
    xyz = [p[:3, 3] for p in poses]
    dist = [0]
    for i in range(1, len(poses)):
        d = dist[i - 1] + np.linalg.norm(xyz[i] - xyz[i - 1])
        dist.append(d)
    return dist


def last_frame_from_segment_length(dist, first_frame, length):
    """Find frame (index) that away from the first_frame with
    the required distance
    Args:
        dist (float list): distance of each pose w.r.t frame-0
        first_frame (int): start-frame index
        length (float): required distance
    Returns:
        i (int) / -1: end-frame index. if not found return -1
    """
    for i in range(first_frame, len(dist), 1):
        if dist[i] > (dist[first_frame] + length):
            return i
    return -1


def rotation_error(pose_error):
    """Compute rotation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        rot_error (float): rotation error
    """
    a = pose_error[0, 0]
    b = pose_error[1, 1]
    c = pose_error[2, 2]
    d = 0.5 * (a + b + c - 1.0)
    rot_error = np.arccos(max(min(d, 1.0), -1.0))
    return rot_error


def translation_error(pose_error):
    """Compute translation error
    Args:
        pose_error (4x4 array): relative pose error
    Returns:
        trans_error (float): translation error
    """
    dx = pose_error[0, 3]
    dy = pose_error[1, 3]
    dz = pose_error[2, 3]
    trans_error = np.sqrt(dx**2 + dy**2 + dz**2)
    return trans_error


def calc_sequence_errors(pred_poses, gt_poses):
    """calculate sequence error
    """
    error = []
    dist = trajectory_distances(gt_poses)
    step_size = 10
    sequence_lengths = [100, 200, 300, 400, 500, 600, 700, 800]
    num_lengths = len(sequence_lengths)

    for first_frame in range(0, len(gt_poses), step_size):
        for i in range(num_lengths):
            length = sequence_lengths[i]
            last_frame = last_frame_from_segment_length(dist, first_frame, length)

            # Continue if sequence is not long enough
            if last_frame == -1:
                continue

            # Compute rotational and translational errors
            pose_delta_gt = np.linalg.inv(gt_poses[first_frame]) @ gt_poses[last_frame]
            pose_delta_pred = np.linalg.inv(pred_poses[first_frame]) @ pred_poses[last_frame]
            pose_error = np.linalg.inv(pose_delta_pred) @ pose_delta_gt
            rot_error = rotation_error(pose_error) / length
            trans_error = translation_error(pose_error) / length

            # compute speed
            num_frames = last_frame - first_frame + 1
            speed = length / (0.1 * num_frames)  # Assume 10fps

            error.append([first_frame, rot_error, trans_error, length, speed])
    return error


def compute_segment_error(seq_errs):
    """This function calculates average errors for different segment.
    Args:
        seq_errs (list list): list of errs; [first_frame, rotation error, translation error,
                                             length, speed]
            - first_frame: frist frame index
            - rotation error: rotation error per length
            - translation error: translation error per length
            - length: evaluation trajectory length
            - speed: car speed (#FIXME: 10FPS is assumed)
    Returns:
        avg_segment_errs (dict): {100:[avg_t_err, avg_r_err],...}
    """

    sequence_lengths = [100, 200, 300, 400, 500, 600, 700, 800]

    segment_errs = {}
    avg_segment_errs = {}
    for len_ in sequence_lengths:
        segment_errs[len_] = []

    # Get errors
    for err in seq_errs:
        len_ = err[3]
        t_err = err[2]
        r_err = err[1]
        segment_errs[len_].append([t_err, r_err])

    # Compute average
    for len_ in sequence_lengths:
        if segment_errs[len_] != []:
            avg_t_err = np.mean(np.asarray(segment_errs[len_])[:, 0])
            avg_r_err = np.mean(np.asarray(segment_errs[len_])[:, 1])
            avg_segment_errs[len_] = [avg_t_err, avg_r_err]
        else:
            avg_segment_errs[len_] = []
    return avg_segment_errs


def compute_overall_err(seq_err):
    """Compute average translation & rotation errors
    Args:
        seq_err (list list): [[r_err, t_err],[r_err, t_err],...]
            - r_err (float): rotation error
            - t_err (float): translation error
    Returns:
        ave_t_err (float): average translation error
        ave_r_err (float): average rotation error
    """
    t_err = 0
    r_err = 0

    seq_len = len(seq_err)

    if seq_len == 0:
        return 0, 0
    for item in seq_err:
        r_err += item[1]
        t_err += item[2]
    ave_t_err = t_err / seq_len
    ave_r_err = r_err / seq_len
    return ave_t_err, ave_r_err


def compute_ATE(pred_poses, gt_poses):
    """Compute RMSE of ATE (abs. trajectory error)
    """
    errors = []
    for pred_pose, gt_pose in zip(pred_poses, gt_poses):
        gt_xyz = gt_pose[:3, 3]
        pred_xyz = pred_pose[:3, 3]
        align_err = gt_xyz - pred_xyz

        errors.append(np.sqrt(np.sum(align_err**2)))
    ate = np.sqrt(np.mean(np.asarray(errors)**2))
    return ate


def compute_RPE(pred_poses, gt_poses):
    """Compute RPE (rel. pose error)
    Returns:
        rpe_trans
        rpe_rot
    """
    trans_errors = []
    rot_errors = []
    for i in range(len(pred_poses) - 1):
        # for i in list(pred.keys())[:-1]:
        gt1 = gt_poses[i]
        gt2 = gt_poses[i + 1]
        gt_rel = np.linalg.inv(gt1) @ gt2

        pred1 = pred_poses[i]
        pred2 = pred_poses[i + 1]
        pred_rel = np.linalg.inv(pred1) @ pred2

        rel_err = np.linalg.inv(gt_rel) @ pred_rel
        trans_errors.append(translation_error(rel_err))
        rot_errors.append(rotation_error(rel_err))
    rpe_trans = np.mean(np.asarray(trans_errors))
    rpe_rot = np.mean(np.asarray(rot_errors))
    return rpe_trans, rpe_rot


def calc_error(pred_poses, gt_poses, optimize_scale: bool = False) -> str:
    log = ''
    if optimize_scale:
        pred_poses_scaled, scaling = scale_optimization(pred_poses, gt_poses)
        log += '-' * 10 + ' MEDIAN\n'
        log += f'Scaling: {scaling}'
    else:
        pred_poses_scaled = pred_poses
    sequence_error = calc_sequence_errors(pred_poses_scaled, gt_poses)
    # segment_error = compute_segment_error(sequence_error)
    ave_t_err, ave_r_err = compute_overall_err(sequence_error)

    log += '-' * 10 + '\n'
    log += f'Trans error (%):      {ave_t_err * 100:.4f}' + '\n'
    log += f'Rot error (deg/100m): {100 * ave_r_err / np.pi * 180:.4f}' + '\n'

    # Compute ATE
    ate = compute_ATE(pred_poses, gt_poses)
    log += f'Abs traj RMSE (m):    {ate:.4f}' + '\n'

    # Compute RPE
    rpe_trans, rpe_rot = compute_RPE(pred_poses, gt_poses)
    log += f'Rel pose error (m):   {rpe_trans:.4f}' + '\n'
    log += f'Rel pose err (deg):   {rpe_rot * 180 / np.pi:.4f}' + '\n'
    log += '-' * 10 + '\n'

    return log


# =============================================================================


def calc_depth_error(
    pred_depth,
    gt_depth,
    median_scaling: bool = True,
    min_depth: Optional[float] = None,
    max_depth: Optional[float] = None,
) -> Dict[str, float]:
    gt_height, gt_width = gt_depth.shape
    pred_depth = cv2.resize(pred_depth, (gt_width, gt_height))

    # Mask out pixels without ground truth depth
    # or ground truth depth farther away than the maximum predicted depth
    if max_depth is not None:
        mask = np.logical_and(gt_depth > min_depth, gt_depth < max_depth)
    else:
        mask = gt_depth > min_depth
    pred_depth = pred_depth[mask]
    gt_depth = gt_depth[mask]

    # Introduced by SfMLearner
    if median_scaling:
        ratio = np.median(gt_depth) / np.median(pred_depth)
        pred_depth *= ratio

    # Cap predicted depth at min and max depth
    pred_depth[pred_depth < min_depth] = min_depth
    if max_depth is not None:
        pred_depth[pred_depth > max_depth] = max_depth

    # Compute error metrics
    thresh = np.maximum((gt_depth / pred_depth), (pred_depth / gt_depth))
    a1 = np.mean(thresh < 1.25)
    a2 = np.mean(thresh < 1.25**2)
    a3 = np.mean(thresh < 1.25**3)
    rmse = (gt_depth - pred_depth)**2
    rmse_tot = np.sqrt(np.mean(rmse))
    rmse_log = (np.log(gt_depth) - np.log(pred_depth))**2
    rmse_log_tot = np.sqrt(np.mean(rmse_log))
    abs_diff = np.mean(np.abs(gt_depth - pred_depth))
    abs_rel = np.mean(np.abs(gt_depth - pred_depth) / gt_depth)
    sq_rel = np.mean(((gt_depth - pred_depth)**2) / gt_depth)

    metrics = {
        'abs_diff': abs_diff,
        'abs_rel': abs_rel,
        'sq_rel': sq_rel,
        'a1': a1,
        'a2': a2,
        'a3': a3,
        'rmse': rmse_tot,
        'rmse_log': rmse_log_tot
    }

    return metrics
