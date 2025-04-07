from mujoco.mjx._src.types import JointType
from dataclasses import dataclass
from typing import Callable, Optional, Set
import jax
import jax.numpy as jnp
import numpy as np
from jax.flatten_util import ravel_pytree
from jax._src.util import unzip2
from mujoco import mjx

@jax.tree_util.register_static
@dataclass(frozen=True)
class FDCache:
    """Holds all the precomputed info needed by the custom FD-based backward pass."""
    unravel_dx: Callable[[jnp.ndarray], mjx.Data]
    num_u_dims: int
    eps: float
    sensitivity_mask: jnp.ndarray
    dx_flat_all_idx: Optional[jnp.ndarray]  # All indices (including free/ball quaternion)
    dx_flat_no_quat_idx: jnp.ndarray  # "normal" FD indices (excludes free/ball quaternion)
    dx_flat_quat_idx: Optional[jnp.ndarray] = None # subset that also lies in target_fields
    dx_flat_init_quat_idx: Optional[jnp.ndarray] = None # first index of each quaternion
    qpos_init_quat_idx_rep: Optional[jnp.ndarray] = None # repeated by ijk axes * quat joints
    quat_ijk_idx_rep: Optional[jnp.ndarray] = None # repeated by ijk axes * quat joints

def build_fd_cache(
    mx,                   # MuJoCo model wrapper, for jnt_type, jnt_qposadr
    dx_ref: mjx.Data,     # reference data object
    target_fields: Optional[Set[str]] = None,
    ctrl_dim: Optional[int] = None,
    eps: float = 1e-6
) -> FDCache:
    """
    Build an FDCache that:
      1) Finds the global flatten range of qpos within dx_ref.
      2) Converts MuJoCo's local qpos indices for free/ball joints
         into *global* flatten indices (quat_idx).
      3) Gathers all target_fields into 'candidate_idx_full'.
      4) Splits out 'quat_inner_idx' from 'candidate_idx_full', removing them
         so 'inner_idx' contains only non-quaternion states.
    """
    if target_fields is None:
        target_fields = {"qpos", "qvel"}

    # Flatten dx_ref
    dx_array, unravel_dx = ravel_pytree(dx_ref)
    dx_size = dx_array.shape[0]

    # If ctrl_dim not specified, get from dx_ref.ctrl
    if ctrl_dim is None:
        ctrl_dim = dx_ref.ctrl.shape[0]
    num_u_dims = ctrl_dim

    # ----------------------------------------------------------------
    # A) Identify the global flatten range for the "qpos" leaf
    # ----------------------------------------------------------------
    # We assume there's exactly one leaf whose path has name=='qpos'
    leaves_with_path = list(jax.tree_util.tree_leaves_with_path(dx_ref))
    sizes, _ = unzip2((jnp.size(leaf), jnp.shape(leaf)) for (_, leaf) in leaves_with_path)
    offsets = np.cumsum(sizes)  # The end offsets for each leaf in the flatten array

    # We'll find which leaf index i is 'qpos'
    qpos_leaf_idx = None
    running_start = 0
    for i, (path, leaf_val) in enumerate(leaves_with_path):
        leaf_end = offsets[i]
        # Check if this leaf is "qpos"
        name_matches = any(getattr(p, 'name', None) == 'qpos' for p in path)
        if name_matches:
            qpos_leaf_idx = i
            qpos_leaf_start = running_start
            break
        running_start = leaf_end

    # If we can't find qpos leaf, raise an error or handle differently
    if qpos_leaf_idx is None:
        raise RuntimeError("Could not find a 'qpos' leaf in dx_ref to map quaternion indices.")

    # So the global flatten range for qpos is [qpos_leaf_start : qpos_leaf_start + qpos_leaf_size]
    # local qpos index i in [0..(nq-1)] maps to global index (qpos_leaf_start + i)

    # ----------------------------------------------------------------
    # B) Build "quat_idx_full" in global flatten space
    # ----------------------------------------------------------------
    # For each free or ball joint, we add the last 4 or all 4 local qpos indices,
    # then shift them by qpos_leaf_start to get global indices.
    # e.g. for FREE => local [3..6] => global [qpos_leaf_start+3..+6]
    # for BALL => local [0..3], etc.
    local_quat_indices = []
    for j, jtype in enumerate(mx.jnt_type):
        if jtype == JointType.FREE:
            start = mx.jnt_qposadr[j]     # local qpos index for that joint
            local_quat_indices.append(np.arange(start+3, start+7))
        elif jtype == JointType.BALL:
            start = mx.jnt_qposadr[j]
            local_quat_indices.append(np.arange(start, start+4))
        # else HINGE/SLIDE => skip

    if len(local_quat_indices) == 0:
        quat_idx_full = np.array([], dtype=int)
        quat_idx = None
    else:
        # flatten them
        loc_concat = np.concatenate(local_quat_indices, axis=0)
        quat_idx = jnp.array(loc_concat, dtype=jnp.int32)
        # shift by qpos_leaf_start
        quat_idx_full = loc_concat + qpos_leaf_start

    # ----------------------------------------------------------------
    # C) Gather "candidate_idx_full" for target_fields
    #     i.e. flatten indices for qpos, qvel, ctrl, etc.
    # ----------------------------------------------------------------
    def leaf_index_range(leaf_idx):
        # [start, end)
        start_ = 0 if leaf_idx == 0 else offsets[leaf_idx-1]
        end_ = offsets[leaf_idx]
        return np.arange(start_, end_)

    candidate_subsets = []
    for i, (path, leaf_val) in enumerate(leaves_with_path):
        name_matches = any(getattr(p, 'name', None) in target_fields for p in path)
        if name_matches:
            candidate_subsets.append(leaf_index_range(i))
    if len(candidate_subsets) > 0:
        dx_flat_all_idx = np.concatenate(candidate_subsets, axis=0)
    else:
        dx_flat_all_idx = np.array([], dtype=int)

    # ----------------------------------------------------------------
    # D) Build "dx_flat_quat_idx" = intersection of dx_flat_all_idx & quat_idx_full
    #    Then remove them from dx_flat_all_idx => "non_quat_inner_idx"
    # ----------------------------------------------------------------
    if quat_idx_full.size == 0 or dx_flat_all_idx.size == 0:
        quat_inner = np.array([], dtype=int)
    else:
        # intersect
        quat_inner = np.intersect1d(quat_idx_full, dx_flat_all_idx)

    # Remove them from candidate
    non_quat_inner = np.setdiff1d(dx_flat_all_idx, quat_inner)

    # Convert to jnp
    if quat_inner.size == 0:
        dx_flat_quat_idx = np.array([], dtype=int)
    else:
        dx_flat_quat_idx = jnp.array(quat_inner, dtype=jnp.int32)

    dx_flat_no_quat_idx = jnp.array(non_quat_inner, dtype=jnp.int32)

    # ----------------------------------------------------------------
    # E) Build sensitivity_mask for the final "dx_flat_no_quat_idx" only
    # ----------------------------------------------------------------
    sensitivity_mask = jnp.zeros_like(dx_array)
    sensitivity_mask = sensitivity_mask.at[dx_flat_no_quat_idx].set(1.0)
    sensitivity_mask = sensitivity_mask.at[dx_flat_quat_idx].set(1.0)

    # quat by inner size by 2
    # get befginning indexes of the quats
    if dx_flat_quat_idx.size != 0 and quat_idx.size != 0:
        dx_flat_init_quat_idx = jnp.repeat(dx_flat_quat_idx[::4],3)
        qpos_init_quat_idx_rep = jnp.repeat(quat_idx[::4],3)
    else:
        dx_flat_init_quat_idx = jnp.array([], dtype=jnp.int32)
        qpos_init_quat_idx_rep = jnp.array([], dtype=jnp.int32)

    # FOr each index above,
    quat_ijk_idx_rep = jnp.tile(jnp.array([0, 1, 2], dtype=jnp.int32), len(dx_flat_quat_idx) // 4)

    # ----------------------------------------------------------------
    # F) Return FDCache
    # ----------------------------------------------------------------
    return FDCache(
        unravel_dx = unravel_dx,
        sensitivity_mask = sensitivity_mask,
        dx_flat_no_quat_idx= dx_flat_no_quat_idx,
        dx_flat_all_idx= dx_flat_all_idx,
        num_u_dims = num_u_dims,
        eps = eps,
        qpos_init_quat_idx_rep= qpos_init_quat_idx_rep,
        dx_flat_quat_idx= dx_flat_quat_idx,
        dx_flat_init_quat_idx= dx_flat_init_quat_idx,
        quat_ijk_idx_rep= quat_ijk_idx_rep
    )
