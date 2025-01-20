import equinox as eqx
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

def clip_grad_elementwise(grads, clip_value=1.0):
    """
    Clips each element of the gradients PyTree to lie within [-clip_value, clip_value].

    Args:
        grads: A PyTree containing gradient arrays.
        clip_value: The maximum absolute value for each gradient element.

    Returns:
        A PyTree with clipped gradients.
    """
    return jax.tree_util.tree_map(lambda g: jnp.clip(g, -clip_value, clip_value), grads)


def clip_grad_global_norm(grads, max_norm=1.0):
    """
    Clips the gradients PyTree based on the global norm.

    Args:
        grads: A PyTree containing gradient arrays.
        max_norm: The maximum allowed norm for the gradients.

    Returns:
        A PyTree with scaled gradients if the global norm exceeds max_norm.
    """
    global_norm = jnp.sqrt(jnp.sum(jax.tree_util.tree_map(lambda g: jnp.sum(g ** 2), grads)))
    scaling_factor = jnp.minimum(1.0, max_norm / (global_norm + 1e-6))
    return jax.tree_util.tree_map(lambda g: g * scaling_factor, grads)


class Network(eqx.Module, ABC):
    """
    Abstract base class for policies. Users should inherit from this class
    and implement the __call__ method.
    """

    @abstractmethod
    def __call__(self, x, t):
        """
        Forward pass of the network.

        Args:
            x (jnp.ndarray): Input features.
            t (jnp.ndarray): Additional temporal or contextual information.

        Returns:
            jnp.ndarray: Network output.
        """
        pass

    @staticmethod
    @eqx.filter_jit
    def make_step(dxs, optim, model, state, ctx, user_key):
        """
        Performs a single optimization step.

        Args:
            dxs: ..
            optim: Optimizer instance (e.g., from optax).
            model (BasePolicy): The model to update.
            state: Optimizer state.
            ctx: Context object containing additional information like loss function.

        Returns:
            Tuple[BasePolicy, state, float]: Updated model, updated state, and loss value.
        """
        params, static = eqx.partition(model, eqx.is_array)
        (loss_value, res), grads = jax.value_and_grad(ctx.cbs.loss_func, has_aux=True)(
            params, static, dxs, ctx, user_key
        )
        # grads = jax.tree_util.tree_map(lambda x: jnp.nan_to_num(x), grads)
        # grads = clip_grad_elementwise(grads, clip_value=1.0)

        updates, state = optim.update(grads, state, model)
        model = eqx.apply_updates(model, updates)

        return model, state, loss_value, res

    @staticmethod
    @eqx.filter_jit
    # TODO with dxs
    def make_step_multi_gpu(optim, model, state, x_init, ctx, user_key):
        """
        Performs a single optimization step on multiple GPUs.

        Args:
            optim: Optimizer instance (e.g., from optax).
            model (BasePolicy): The model to update.
            state: Optimizer state.
            x_init: Initial input data.
            ctx: Context object containing additional information like loss function.
        """
        raise NotImplementedError
        params, static = eqx.partition(model, eqx.is_array)

        # Reshape x_init to have leading dimension equal to the number of GPUs
        num_devices = ctx.cfg.num_gpu
        x_init = x_init.reshape(num_devices, -1, x_init.shape[-1])

        # Define the function to be pmapped
        def per_device_loss_and_grad(params, static, x_init, ctx):
            # Compute per-device loss and gradient
            (loss_value, traj_costs), grads = jax.value_and_grad(ctx.cbs.loss_func, has_aux=True)(
                params, static, x_init, ctx, user_key
            )
            # Average loss and gradients across devices
            grads = jax.lax.pmean(grads, axis_name='devices')
            return (loss_value, traj_costs), grads

        # Compute loss and gradients using pmap
        (loss_value, traj_costs), grads = eqx.filter_pmap(
            per_device_loss_and_grad,
            axis_name='devices',
            in_axes=(None, None, 0, None),
        )(params, static, x_init, ctx)

        # Extract the averaged gradients from one device
        grads = jax.tree_util.tree_map(lambda x: x[0], grads)

        # Perform optimization step on a single device
        updates, state = optim.update(grads, state, model)
        model = eqx.apply_updates(model, updates)

        return model, state, loss_value[0], traj_costs[0]