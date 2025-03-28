import jax
import jax.numpy as jnp
from mujoco.mjx._src.dataclasses import PyTreeNode


class DummyData(PyTreeNode):
    qpos: jax.Array = jnp.array([1.5,0.,3.])
    qacc_smooth: jax.Array = jnp.array([1.5,0.,3.])


def basis_like(x):
    shape_ = x.shape
    size_  = jnp.prod(jnp.array(shape_))
    eye_   = jnp.eye(size_, dtype=x.dtype)
    return eye_.reshape((size_,) + shape_)

def fd_qacc_smooth(qacc_smooth_dir):
    # delta = build_perturbation(qacc_smooth_dir)
    return DummyData()

dx = DummyData()
dx_out_bar = DummyData()

qacc_smooth_bases = basis_like(dx.qacc_smooth)
qacc_smooth_sensitivity = jax.vmap(fd_qacc_smooth)(qacc_smooth_bases)

qacc_smooth = jax.tree.reduce(lambda acc,x: acc + x, jax.tree_util.tree_map(lambda g,x: jnp.dot(x,g), dx_out_bar, qacc_smooth_sensitivity))


dx_out_bar = jax.tree_util.tree_map(jnp.zeros_like, dx_out_bar)
# g_out = jax.tree_util.tree_map(jnp.zeros_like, dx)
dx_out_bar = dx_out_bar.replace(qacc_smooth=qacc_smooth)
