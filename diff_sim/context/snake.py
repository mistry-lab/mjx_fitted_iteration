import os
import jax
from jax import numpy as jnp
import equinox as eqx
import mujoco
from mujoco import mjx
from diff_sim.loss_funcs import loss_fn_policy_stoch, loss_fn_td_stoch, loss_fn_td_det, loss_fn_policy_det
from diff_sim.context.meta_context import Config, Callbacks, Context
from diff_sim.nn.base_nn import Network

model_path = os.path.join(os.path.dirname(__file__), '../xmls/snake_mass.xml')
def gen_model() -> mujoco.MjModel:
    return mujoco.MjModel.from_xml_path(model_path)


def parse_sensordata(name, mx, dx):
    id = mjx.name2id(mx, mujoco.mjtObj.mjOBJ_SENSOR, name)
    i = mx.sensor_adr[id]
    dim = mx.sensor_dim[id]
    return dx.sensordata[i:i+dim]

_cfg = Config(
    lr=5e-3,
    seed=4,
    batch=512,
    samples=1,
    epochs=1000,
    eval=50,
    num_gpu=1,
    dt=0.01,
    ntotal=512,
    nsteps=32,
    mx=mjx.put_model(gen_model()),
    gen_model=gen_model
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
        x = self.layers[-1](x).squeeze()
        # x = jnp.tanh(x) * 2.
        # x_arm = jnp.tanh(x[:3]) * 1.
        control = jnp.concatenate([jnp.tanh(x[:2]) * 1.2, # accx , accy forces
                                   jnp.array([jnp.tanh(x[2]) * 0.002]), # accz forces
                                   jnp.tanh(x[3:6]) * 0.01],  # angular forces
                                #    jnp.tanh(x[6:]) * 1.],   # Revolute joints
                                   axis=0)
        # x = jnp.concatenate([x_arm, ball_control * 0, jnp.zeros(3)], axis=0)
        return control

def policy(net: Network, mx: mjx.Model, dx: mjx.Data, policy_key: jnp.ndarray
) -> tuple[mjx.Data, jnp.ndarray]:
    x = state_encoder(mx, dx)
    t = jnp.expand_dims(dx.time, axis=0)
    u = net(x, t)
    # 6 First on middle body
    dx = dx.replace(qfrc_applied=dx.qfrc_applied.at[:6].set(u[:6]))
    # 2 Revolute joint.
    # dx = dx.replace(ctrl=dx.ctrl.at[:].set(u[6:]))
    return dx, u

def state_encoder(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    ball_pos = dx.qpos[7:9]
    # mocap pos for some reason has one leading dimension (inherent to mujoco)
    pos_diff = ball_pos - dx.mocap_pos[0, :2]
    # joint_pos = dx.qpos[:3]  # 2
    # middle_pos = dx.qpos[:3] 
    # joint_vel = dx.qvel[:3]
    ball_vel = dx.qvel[6:8]
    qpos = dx.qpos[:-4]
    dcom = dx.xipos[1,:2] - ball_pos

    return jnp.concatenate([qpos, dcom, pos_diff, ball_vel], axis=0)

def state_decoder(x: jnp.ndarray) -> jnp.ndarray:
    return x

def control_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    x = dx.qfrc_applied[:6]
    return jnp.dot(
        x.T, jnp.dot(jnp.diag(jnp.array([0.0001, 0.0001, 0.0001, 0.0001, 0.0001, 0.001])), x)
    )

def run_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    x = state_encoder(mx, dx)[10:14]
    # bound_cost =jnp.maximum(jnp.abs(x[0]) - 2*jnp.pi, 0)
    state_cost =  _cfg.dt * jnp.dot(
        x.T, jnp.dot(jnp.diag(jnp.array([1000., 1000., 1000., 1000.])), x)
    )

    # touch = parse_sensordata("touch_sphere", mx, dx).squeeze()
    # threashold = 0.1
    # threashold contact detected.
    # touch_cost = state_cost * (1 / threashold) * jnp.maximum(threashold - touch, 0.)

    # touch_cost2 = state_cost2 * (1 / threashold) * jnp.maximum(touch - threashold, 0.)

    
    return state_cost   

def terminal_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    x = state_encoder(mx, dx)[10:14]
    # bound_cost =jnp.maximum(jnp.abs(x[0]) - 2*jnp.pi, 0)
    state_cost =  _cfg.dt * jnp.dot(
        x.T, jnp.dot(jnp.diag(jnp.array([1000., 1000., 1000., 1000.])), x)
    )
    # state_cost2 = _cfg.dt * jnp.dot(
    #     x.T, jnp.dot(jnp.diag(jnp.array([1000., 1000.])), x)
    # )

    # touch = parse_sensordata("touch_sphere", mx, dx).squeeze()
    # threashold = 0.1
    # # threashold contact detected.
    # touch_cost = state_cost * (1 / threashold) * jnp.maximum(threashold - touch, 0.)

    # touch_cost2 = state_cost2 * (1 / threashold) * jnp.maximum(touch - threashold, 0.)

    
    return state_cost  

# def terminal_cost(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
#     x = state_encoder(mx, dx)
#     return _cfg.dt * jnp.dot(
#         x.T, jnp.dot(jnp.diag(jnp.array([0, 0, 0, 100, 100,1000.,1000., 0, 0, .0, .0, .0, 0, 0])), x)
#     )

def set_data(mx: mjx.Model, dx: mjx.Data, ctx: Context, key: jnp.ndarray) -> mjx.Data:
    _, key = jax.random.split(key)
    qvel = jax.random.uniform(key, (_cfg.mx.nv,), minval=-0.01, maxval=0.01)
    _, key = jax.random.split(key)
    ball_pos = jax.random.uniform(key, (2,), minval=-0.4, maxval=0.4)
    _, key = jax.random.split(key)
    target_pos = jax.random.uniform(key, (2,), minval=-0.4, maxval=0.4)
    _, key = jax.random.split(key)
    middle_pos = jax.random.uniform(key, (2,), minval=-0.4, maxval=0.4)

    qpos = jnp.concatenate([middle_pos, jnp.array([0.035]), jnp.array([0.707107, 0.707107, 0, 0]), ball_pos, jnp.array([0.035]), jnp.array([1, 0, 0, 0])], axis=0)

    qpos = dx.qpos.at[:].set(qpos)
    # qpos = dx.qpos.at[:2].set(middle_pos)
    # qpos = dx.qpos.at[7:9].set(ball_pos)
    qvel = dx.qvel.at[:].set(qvel)
    mocap_pos = dx.mocap_pos.at[0,:2].set(target_pos)
    return dx.replace(qpos=qpos, qvel=qvel, mocap_pos=mocap_pos)
 
def gen_network(seed: int) -> Network:
    key = jax.random.PRNGKey(seed)
    return Policy([16, 128, 128, 6], key)

def is_terminal(mx: mjx.Model, dx: mjx.Data) -> jnp.ndarray:
    time_limit =  (dx.time/ mx.opt.timestep) > (_cfg.ntotal - 1)
    return jnp.array([time_limit])


ctx = Context(
    _cfg,
    Callbacks(
        run_cost=run_cost,
        terminal_cost=terminal_cost,
        control_cost=control_cost,
        set_data=set_data,
        state_encoder=state_encoder,
        state_decoder=state_decoder,
        gen_network=gen_network,
        controller=policy,
        loss_func=loss_fn_policy_det,
        is_terminal=is_terminal
    )
)


