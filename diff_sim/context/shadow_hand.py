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
from diff_sim.utils.math_helper import quaternion_difference, random_quaternion

model_path = os.path.join(os.path.dirname(__file__), '../xmls/shadow_hand/scene_right.xml')

# TODO:
def gen_model() -> mujoco.MjModel:
    m = mujoco.MjModel.from_xml_path(model_path)
    # Modify reference position. (qpos - qpos0)
    m.qpos0[:24] = -np.array([
        -0.056, 0.014, -0.077, 0.55, 0.91, 1.1, 0.052, 0.7, 1, 0.54,
        0.062, 0.59, 1.1, 0.52, 0.22, -0.081, 0.39, 1.1, 0.99, -0.24,
        0.63, 0.2, 0.7, 0.65])
    
    return m
    
_cfg = Config(
        lr=4e-3,
        num_gpu=1,
        seed=0,
        nsteps=24,
        ntotal=800,
        epochs=1000,
        batch=2,
        samples=1,
        eval=10,
        dt=0.005,
        mx= mjx.put_model(gen_model()),
        gen_model=gen_model,
    )

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
        # t = t if t.ndim == 1 else t.reshape(1)
        # x = jnp.concatenate([x, t], axis=-1)
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x).squeeze()


# TODO: Remove context from anything that is defined in this file as anything s
def policy(net: Network, mx: mjx.Model, dx: mjx.Data, policy_key: jnp.ndarray
) -> tuple[mjx.Data, jnp.ndarray]:
    x = state_encoder(mx, dx)
    t = jnp.expand_dims(dx.time, axis=0)
    u = net(x, t)
    # u = 0.5*net(x, t)
    # u += 0.002*jax.random.normal(policy_key, u.shape)
    # Setup offset
    dx = dx.replace(ctrl=dx.ctrl.at[:].set(u))
    return dx, u

def parse_sensordata(name, mx, dx):
    id = mjx.name2id(mx, mujoco.mjtObj.mjOBJ_SENSOR, name)
    i = mx.sensor_adr[id]
    dim = mx.sensor_dim[id]
    return dx.sensordata[i:i+dim]

def run_cost(mx: mjx.Model,dx:mjx.Data) -> jnp.ndarray:
    """ 
    Running costs. Number of costs: 5
    (1): object_position - palm_position
    (2): object_orientation - goal_orientation
    (3): Object linear velocity
    (4): Object angular velocity
    (5): hand joint velocity
    """
    pos = parse_sensordata("object_position", mx, dx)
    pos_ref = parse_sensordata("palm_position", mx, dx)
    cpos = 0.1*jnp.sum((pos - pos_ref)**2)

    quat = parse_sensordata("object_orientation", mx, dx)
    quat_ref = parse_sensordata("goal_orientation", mx, dx)
    # quat_ref = jnp.array([1.,0.,0.,0.])
    cquat = 0.5*jnp.sum(quaternion_difference(quat, quat_ref)**2)

    vel = parse_sensordata("object_linear_velocity", mx, dx)
    cvel = 0.05*jnp.sum(vel**2)

    ang_vel = parse_sensordata("object_angular_velocity", mx, dx)
    cang_vel = 0.05*jnp.sum(ang_vel**2)

    cjoint_pos = 0.2*jnp.sum(dx.qpos[:24]**2) 
    cjoint_vel = 0.2*jnp.sum(dx.qvel[:24]**2)

    return cpos + cquat + cvel + cang_vel + cjoint_pos + cjoint_vel

def terminal_cost(mx: mjx.Model,dx:mjx.Data) -> jnp.ndarray:
    """
    Terminal costs. Number of costs: 5
    (1): object_position - palm_position
    (2): object_orientation - goal_orientation
    (3): Object linear velocity
    (4): Object angular velocity
    (5): Hand joint position around ref (0 vector)
    (6): Hand joint velocity
    """
    pos = parse_sensordata("object_position", mx, dx)
    pos_ref = parse_sensordata("palm_position", mx, dx)
    cpos = 0.5*jnp.sum((pos - pos_ref)**2)

    quat = parse_sensordata("object_orientation", mx, dx)
    quat_ref = parse_sensordata("goal_orientation", mx, dx)
    # quat_ref = jnp.array([1.,0.,0.,0.])
    cquat = 2.*jnp.sum(quaternion_difference(quat, quat_ref)**2)

    vel = parse_sensordata("object_linear_velocity", mx, dx)
    cvel = 0.25*jnp.sum(vel**2)

    ang_vel = parse_sensordata("object_angular_velocity", mx, dx)
    cang_vel = 0.25*jnp.sum(ang_vel**2)

    cjoint_pos = 0.2*jnp.sum(dx.qpos[:24]**2)    
    cjoint_vel = 0.2*jnp.sum(dx.qvel[:24]**2)

    return cpos + cquat + cvel + cang_vel + cjoint_pos + cjoint_vel

def control_cost(mx: mjx.Model,dx:mjx.Data) -> jnp.ndarray:
    """ Control cost. Due to position control, penalisation 
    of actuator_force instead.
    """
    x = dx.actuator_force
    return jnp.dot(x.T, jnp.dot(jnp.diag(jnp.full((24,), 10.)), x))


def init_gen(total_batch: int, key: jnp.ndarray) -> jnp.ndarray:
    """ Initialise Shadow hand scene. (Right hand, sphere object and sphere goal)
    """
    # 1. Generate joint_pos and joint_vel
    key, subkey1, subkey2, subkey3, subkey4 = jax.random.split(key, 5)  # Splitting key for different random generations
    joint_pos = jax.random.uniform(subkey1, (total_batch, 24), minval=-0.01, maxval=0.01)
    joint_vel = jax.random.uniform(subkey2, (total_batch, 24), minval=-0.05, maxval=0.05)
    object_quat = random_quaternion(subkey3, total_batch)
    # object_quat = jnp.tile(jnp.array([1., 0.0, 0., 0.]), (total_batch, 1))
    goal_quat = random_quaternion(subkey4, total_batch)
    # goal_quat = jnp.tile(jnp.array([1., 0.0, 0., 0.]), (total_batch, 1))

    # 2. Fixed values for object_pos, object_vel, object_ang_vel, goal_ang_vel
    object_pos = jnp.array([0.3, 0.0, 0.065])  # Shape (3,)
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
        goal_quat,  # Shape (total_batch, 4)
        joint_vel,  # Shape (total_batch, 20)
        object_vel_broadcast,  # Shape (total_batch, 3)
        object_ang_vel_broadcast,  # Shape (total_batch, 3)
        goal_ang_vel_broadcast  # Shape (total_batch, 3)
    ], axis=1)

    return xinits


def state_encoder(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    return jnp.concatenate([dx.qpos, dx.qvel], axis=0)


def state_decoder(x: jnp.ndarray) -> jnp.ndarray:
    return x

def gen_network(seed: int) -> Network:
    key = jax.random.PRNGKey(seed)
    return Policy([68, 64, 64, 24], key)

def is_terminal(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    pos = parse_sensordata("object_position", mx, dx)
    return  jnp.array([jnp.logical_or(pos[2] < -0.05, (dx.time / mx.opt.timestep) > (_cfg.ntotal-1))])

# TODO mx should not be needed in this context this means make step should be modified
ctx = Context(
    _cfg,
    Callbacks(
        run_cost=run_cost,
        terminal_cost=terminal_cost,
        control_cost=control_cost,
        init_gen=init_gen,
        state_encoder=state_encoder,
        state_decoder=state_decoder,
        gen_network=gen_network,
        controller=policy,
        loss_func=loss_fn_policy_det,
        is_terminal=is_terminal
    )
)
