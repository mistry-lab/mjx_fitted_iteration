import equinox as eqx
import jax

@eqx.filter_jit
def step_single_gpu(dxs, model, ctx, user_key, state, optim, simulate_fn, loss_fn):
    """
    Performs a single optimization step.
    :param dxs: Data
    :param model: Model
    :param ctx: Context object containing user-defined information
    :param user_key: Random user_key for sub calls
    :param state: Optimizer state
    :param optim: Optimizer instance (e.g., from optax)
    :param simulate_fn: Function to simulate the model
    :param loss_fn: Loss function
    :return: Updated model, updated state, and loss value
    """
    (loss_value, res), grads = eqx.filter_value_and_grad(loss_fn, has_aux=True)(
        model, dxs, user_key, simulate_fn
    )
    updates, state = optim.update(grads, state, model)
    model = eqx.apply_updates(model, updates)

    return model, state, loss_value, res

@eqx.filter_jit
def step_multi_gpu(dxs, model, ctx, user_key, state, optim, simulate_fn, loss_fn, inner_time, outer_time):
    """
    Performs a single optimization step.
    :param dxs: Data
    :param model: Model
    :param ctx: Context object containing user-defined information
    :param user_key: Random user_key for sub calls
    :param state: Optimizer state
    :param optim: Optimizer instance (e.g., from optax)
    :param simulate_fn: Function to simulate the model
    :param loss_fn: Loss function
    :return: Updated model, updated state, and loss value
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