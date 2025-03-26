import equinox as eqx
import jax
import jax.numpy as jnp

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

@eqx.filter_jit
def step_mppi(dxs, models, ctx, user_key, states, optim, simulate_fn, loss_fns):
    # Simulate
    dxs, x, u, costs, _, terminated = simulate_fn(dxs, user_key, models) #shape: (B, T, 1)
    loss_p_fn, loss_v_fn = loss_fns
    model_p, model_v = models
    state_p, state_v = states

    # Fit policy network
    key_p, key_v, user_key = jax.random.split(user_key, num=3)
    loss_p, grads_p = eqx.filter_value_and_grad(loss_p_fn)(model_p, x, u, key_p)
    # loss_v, grads_v = eqx.filter_value_and_grad(loss_v_fn)(model_v, x, costs, key_v)
    
    # Update policy
    update_p, state_p = optim.update(grads_p, state_p, model_p)
    model_p = eqx.apply_updates(model_p, update_p)
    # Update value
    # update_v, state_v = optim.update(grads_v, state_v, model_v)
    # model_v = eqx.apply_updates(model_v, update_v)

    return (model_p, model_v), (state_p, state_v), (loss_p, 0.), (costs, dxs, terminated, x)

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
        model, dxs, user_key, ctx, simulate_fn
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