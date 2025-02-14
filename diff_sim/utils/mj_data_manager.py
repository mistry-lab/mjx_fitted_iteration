from typing import Callable
from dataclasses import field
from mujoco import mjx
import equinox as eqx
import jax
import jax.numpy as jnp

from diff_sim.context.meta_context import Context

class DataManager(eqx.Module):
    _set_init_compiled: Callable[[mjx.Model, Context, int, jnp.ndarray], mjx.Data] = field(default=None)
    _replace_indices_compiled: Callable[[mjx.Data, jnp.ndarray, mjx.Data], mjx.Data] = field(default=None)

    def __init__(self, set_init, replace_indices):
        self._set_init_compiled = set_init
        self._replace_indices_compiled = replace_indices


    def create_data(
            self, ctx: Context, key: jnp.ndarray
    ) -> mjx.Data:
        dxs = self._set_init_compiled(ctx, key)
        return dxs

    def reset_data(
            self, mx: mjx.Model, dxs: mjx.Data, ctx: Context, key: jnp.ndarray, terminated: jnp.ndarray
    ) -> mjx.Data:
        indices_to_reset = jnp.where(terminated)[0]
        if indices_to_reset.size > 0:
            new_dxs = self.create_data(mx, ctx, indices_to_reset.size, key)
            dxs = self._replace_indices_compiled(dxs, indices_to_reset, new_dxs)
        return dxs

def create_data_manager() -> DataManager:
    def set_init(ctx: Context, key: jnp.ndarray) -> mjx.Data:
        mx = ctx.mx
        batch_size = ctx.batch * ctx.samples
        keys = jax.random.split(key, batch_size)
        dxs = jax.vmap(lambda x: mjx.make_data(mx), in_axes=(0,))(jnp.arange(batch_size))
        dxs = jax.vmap(lambda dx, subkey: ctx.set_data(mx, dx, subkey))(dxs, keys)
        dxs = jax.vmap(lambda dx: mjx.step(mx, dx))(dxs)

        return dxs

    def replace_indices(data: mjx.Data, indices: jnp.ndarray, new_data: mjx.Data) -> mjx.Data:
        def process_field(field, new_field):
            return field.at[indices].set(new_field)

        return jax.tree_util.tree_map(process_field, data, new_data)

    return DataManager(
        eqx.filter_jit(set_init),  eqx.filter_jit(replace_indices)
    )