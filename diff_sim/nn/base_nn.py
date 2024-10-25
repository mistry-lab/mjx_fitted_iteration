import equinox as eqx
import jax
import jax.numpy as jnp
from abc import ABC, abstractmethod

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