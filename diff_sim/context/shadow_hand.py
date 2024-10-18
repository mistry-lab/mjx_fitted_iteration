import os
import jax
import numpy as np
from jax import numpy as jnp
import equinox as eqx
import mujoco
from mujoco import mjx
from diff_sim.loss_funcs import loss_fn_policy_stoch, loss_fn_td_stoch, loss_fn_td_det, loss_fn_policy_det
from diff_sim.context.meta_context import Config, Callbacks, Context
from diff_sim.nn.base_nn import Network

model_path = os.path.join(os.path.dirname(__file__), '../xmls/shadow_hand/scene_right.xml')


class Policy(Network):
    layers: list
    act: callable

    def __init__(self, dims: list, key):
        keys = jax.random.split(key, len(dims))
        self.layers = [eqx.nn.Linear(
            dims[i], dims[i + 1], key=keys[i], use_bias=True
        ) for i in range(len(dims) - 1)]
        self.act = jax.nn.relu

    def __call__(self, x, t):
        t = t if t.ndim == 1 else t.reshape(1)
        x = jnp.concatenate([x, t], axis=-1)
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x).squeeze()


def policy(net: Network, ctx: Context, mx: mjx.Model, dx: mjx.Data, policy_key: jnp.ndarray
) -> tuple[mjx.Data, jnp.ndarray]:
    x = ctx.cbs.state_encoder(mx,dx)
    t = jnp.expand_dims(dx.time, axis=0)
    u = net(x, t)
    dx = dx.replace(ctrl=dx.ctrl.at[:].set(u))
    return dx, u


def run_cost(x: jnp.ndarray) -> jnp.ndarray:
    # x^T Q x
    return jnp.dot(x.T, jnp.dot(jnp.diag(jnp.full((61,), 0.0)), x))


def terminal_cost(x: jnp.ndarray) -> jnp.ndarray:
    # x^T Q_f x
    return 10 * jnp.dot(x.T, jnp.dot(jnp.diag(jnp.full((61,), 0.1)), x))


def control_cost(x: jnp.ndarray) -> jnp.ndarray:
    # u^T R u
    return jnp.dot(x.T, jnp.dot(jnp.diag(jnp.full((24,), 0.01)), x))


def init_gen(total_batch: int, key: jnp.ndarray) -> jnp.ndarray:
    # Generate random quaternions for object_quat and goal_quat (shape (4,))
    def random_quaternion(key, batch_size):
        """Generate a random unit quaternion for each element in the batch."""
        q = jax.random.normal(key, (batch_size, 4))  # Normal distribution for quaternion
        q /= jnp.linalg.norm(q, axis=-1, keepdims=True)  # Normalize to get unit quaternion
        return q

    # 1. Generate joint_pos and joint_vel
    key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 5)  # Splitting key for different random generations
    joint_pos = jax.random.uniform(subkey1, (total_batch, 24), minval=-0.05, maxval=0.05)
    joint_vel = jax.random.uniform(subkey2, (total_batch, 24), minval=-0.05, maxval=0.05)
    # object_quat = random_quaternion(subkey3, total_batch)
    # goal_quat = random_quaternion(subkey4, total_batch)
    object_quat = jnp.tile(jnp.array([1., 0.0, 0., 0.]), (total_batch, 1))
    # goal_quat = jnp.tile(jnp.array([1., 0.0, 0., 0.]), (total_batch, 1))

    # 2. Fixed values for object_pos, object_vel, object_ang_vel, goal_ang_vel
    object_pos = jnp.array([0.3, 0.0, 0.055])  # Shape (3,)
    object_vel = jnp.array([0.0, 0.0, 0.0])  # Shape (3,)
    object_ang_vel = jnp.array([0.0, 0.0, 0.0])  # Shape (3,)
    goal_ang_vel = jnp.array([0.0, 0.0, 0.0])  # Shape (3,)

    object_pos_broadcast = jnp.tile(object_pos, (total_batch, 1))  # Shape (total_batch, 3)
    object_vel_broadcast = jnp.tile(object_vel, (total_batch, 1))  # Shape (total_batch, 3)
    object_ang_vel_broadcast = jnp.tile(object_ang_vel, (total_batch, 1))  # Shape (total_batch, 3)
    goal_ang_vel_broadcast = jnp.tile(goal_ang_vel, (total_batch, 1))  # Shape (total_batch, 3)

    # 5. Concatenate all components along the second axis (axis=1)
    xinits = jnp.concatenate([
        joint_pos,  # Shape (total_batch, 20)
        object_pos_broadcast,  # Shape (total_batch, 3)
        object_quat,  # Shape (total_batch, 4)
        # goal_quat,  # Shape (total_batch, 4)
        joint_vel,  # Shape (total_batch, 20)
        object_vel_broadcast,  # Shape (total_batch, 3)
        object_ang_vel_broadcast,  # Shape (total_batch, 3)
        # goal_ang_vel_broadcast  # Shape (total_batch, 3)
    ], axis=1).squeeze()

    return xinits


def state_encoder(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    return jnp.concatenate([dx.qpos, dx.qvel], axis=0)


def state_decoder(x: jnp.ndarray) -> jnp.ndarray:
    return x

def gen_network(seed: int) -> Network:
    key = jax.random.PRNGKey(seed)
    return Policy([62, 64, 64, 24], key)


def gen_model() -> mujoco.MjModel:
    """ Generate both MjModel and MJX Model.
    """
    m = mujoco.MjModel.from_xml_path(model_path)
    
    # Modify reference position. (qpos - qpos0)
    m.qpos0[:24] = -np.array([
        -0.056, 0.014, -0.077, 0.55, 0.91, 1.1, 0.052, 0.7, 1, 0.54,
        0.062, 0.59, 1.1, 0.52, 0.22, -0.081, 0.39, 1.1, 0.99, -0.24,
        0.63, 0.2, 0.7, 0.65])
    
    return m

ctx = Context(
    Config(
        lr=4e-3,
        num_gpu=1,
        seed=0,
        nsteps=100,
        epochs=1000,
        batch=2,
        samples=1,
        eval=10,
        dt=0.006,
        mx= mjx.put_model(gen_model()),
        gen_model=gen_model,
    ),
    Callbacks(
        run_cost=run_cost,
        terminal_cost=terminal_cost,
        control_cost=control_cost,
        init_gen=init_gen,
        state_encoder=state_encoder,
        state_decoder=state_decoder,
        gen_network=gen_network,
        controller=policy,
        loss_func=loss_fn_policy_det
    )
)
