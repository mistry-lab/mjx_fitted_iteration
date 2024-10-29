from typing import Callable, Tuple
from dataclasses import field
from mujoco import mjx
import equinox as eqx
import jax
import jax.numpy as jnp
from diff_sim.context.meta_context import Context

class DataManager(eqx.Module):
    _set_init_compiled: Callable[[jnp.ndarray, mjx.Model], mjx.Data] = field(default=None)
    _replace_indices_compiled: Callable[[mjx.Data, jnp.ndarray, mjx.Data], mjx.Data] = field(default=None)

    def __init__(self, set_init, replace_indices):
        self._set_init_compiled = set_init
        self._replace_indices_compiled = replace_indices

    def create_data(
            self, mx: mjx.Model, ctx: Context, key: jnp.ndarray
    ) -> Tuple[mjx.Data, jnp.ndarray]:
        x_inits = ctx.cbs.init_gen(ctx.cfg.batch, key)
        dxs = self._set_init_compiled(x_inits, mx)
        return dxs

    def reset_data(
            self, mx: mjx.Model, dxs: mjx.Data, ctx: Context, key: jnp.ndarray, terminated: jnp.ndarray
    ) -> Tuple[mjx.Data, jnp.ndarray]:
        indices_to_reset = jnp.where(terminated)[0]
        if indices_to_reset.size > 0:
            new_dxs = self.create_data(mx, ctx, key)
            dxs = self._replace_indices_compiled(dxs, indices_to_reset, new_dxs)
        return dxs


def create_data_manager() -> DataManager:
    def set_init(x: jnp.ndarray, mx: mjx.Model) -> mjx.Data:
        dx = mjx.make_data(mx)
        qpos = dx.qpos.at[:].set(x[:mx.nq])
        qvel = dx.qvel.at[:].set(x[mx.nq:])
        dx = dx.replace(qpos=qpos, qvel=qvel)
        return mjx.step(mx, dx)

    def replace_indices(data: mjx.Data, indices: jnp.ndarray, new_data: mjx.Data) -> mjx.Data:
        def process_field(field, new_field):
            return field.at[indices].set(new_field)

        return jax.tree_util.tree_map(process_field, data, new_data)

    return DataManager(
        jax.jit(jax.vmap(set_init, in_axes=(0, None))),  jax.jit(replace_indices)
    )