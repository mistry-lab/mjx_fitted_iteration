import os
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')
import jax.numpy as jnp
import equinox as eqx
import mujoco
from mujoco import mjx
from mujoco.mjx._src.math import quat_to_mat, axis_angle_to_quat, quat_to_axis_angle
import optax

import diff_sim
from diff_sim.loss_funcs import loss_fn_policy_det, loss_fn_policy_stoch
from diff_sim.simulation.simulate import make_simulate_fn_fd, make_simulate_fn
from diff_sim.training.train_step import step_single_gpu, step_multi_gpu
from diff_sim.context.meta_context import Context
from diff_sim.utils.mj_data_manager import create_data_manager

from diff_sim.runner_fn import run

if __name__ == "__main__":
    # Load mj and mjx model
    model_path = os.path.join(os.path.dirname(diff_sim.__file__), "xmls", "unitree_a1/task_hill.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    mx = mjx.put_model(model)

    class Policy(eqx.Module):
        layers: list
        act: callable
        dropout: callable

        def __init__(self, dims: list, key):
            keys = jax.random.split(key, len(dims))
            self.layers = [eqx.nn.Linear(
                dims[i], dims[i + 1], key=keys[i], use_bias=True
            ) for i in range(len(dims) - 1)]
            self.act = jax.nn.relu
            self.dropout = eqx.nn.Dropout(0.1)

        def __call__(self, x, key):
            for layer in self.layers[:-1]:
                x = layer(x)
                x = self.act(x)
            x = self.layers[-1](x).squeeze()
            # x = jnp.tanh(x) * 1.
            return x
        
    def set_data(mx: mjx.Model, dx: mjx.Data, key: jnp.ndarray) -> mjx.Data:
        # [front_left_hip, front_left_ankle, front_right_hip ...]
        # q = jnp.zeros(8) 
        # q_hip = jax.random.uniform(key, (4,), minval=-0.5, maxval=0.5) # proximal1
        # q = q.at[::2].set(q_hip)
        quat = jnp.zeros(4)
        quat = quat.at[0].set(1.)

        _, key = jax.random.split(key)
        pos = jnp.zeros(3)
        p_xy =  jax.random.uniform(key, (2,), minval=-0.5, maxval=0.5)
        pos = pos.at[:2].set(p_xy)
        pos = pos.at[2].set(0.28)
        qpos = jnp.concatenate([pos, quat, jnp.zeros(12)])

        _, key = jax.random.split(key)
        v_lin = jax.random.uniform(key, (3,), minval=-0.2, maxval=0.2) # proximal1
        _, key = jax.random.split(key)
        qv = jax.random.uniform(key, (12,), minval=-0.2, maxval=0.2) # proximal1
        qvel = jnp.concatenate([v_lin, jnp.zeros(3), qv])

        dx = dx.replace(qpos=dx.qpos.at[:].set(qpos), qvel=dx.qvel.at[:].set(qvel))
        return dx

    def set_control(dx, u):
        dx = dx.replace(ctrl=dx.ctrl.at[:].set(u))
        return dx

    def gen_network(n: int) -> eqx.Module:
        key = jax.random.PRNGKey(n)
        return Policy([37, 64, 128, 64, 12], key)

    def policy(net: eqx.Module, mx: mjx.Model, dx: mjx.Data, policy_key: jnp.ndarray
    ) -> tuple[mjx.Data, jnp.ndarray]:
        x = jnp.concatenate([dx.qpos, dx.qvel])
        _, key = jax.random.split(policy_key)
        u = net(x, policy_key) + 0.*jax.random.normal(key, shape=(12,))
        return dx, u
    
    # def barrier_cost_quadratic(x, lower=-0.5, upper=0.5, margin=0.1, weight=100.0):
    #     cost_lower = jnp.where(x < lower + margin, ((lower + margin - x) / margin) ** 2, 0.0)
    #     cost_upper = jnp.where(x > upper - margin, ((x - (upper - margin)) / margin) ** 2, 0.0)
    #     return weight * (cost_lower + cost_upper)


    def running_cost(mx: mjx.Model, dx: mjx.Data):
        # quat_ref = axis_angle_to_quat(jnp.array([0.,0.,1.]), jnp.array([2.35]))
        # costR = jnp.sum((quat_to_mat(dx.qpos[4:8])  - quat_to_mat(quat_ref))**2)
        height_reward = (dx.qpos[2] - 0.28)**2
        rot_ang_reward = jnp.sum(dx.qvel[3:6]**2)
        vel_reward = jnp.sum((dx.qvel[0] - jnp.array([1.]))**2)
        ctrl_reward = jnp.sum(dx.ctrl[:]**2)
        # joint_limit_reward = jnp.sum(barrier_cost_quadratic(dx.qpos[7:], lower=-0.5, upper=0.5, margin=0.1, weight=100.0))
        return 0.01*height_reward + 0.*rot_ang_reward + 0.*vel_reward + 0.00001*ctrl_reward 

    def terminal_cost(mx: mjx.Model, dx: mjx.Data):
        height_reward = (dx.qpos[2] - 0.28)**2
        rot_ang_reward = jnp.sum(dx.qvel[3:6]**2)
        vel_reward = jnp.sum((dx.qvel[0] - jnp.array([1.]))**2)
        ctrl_reward = jnp.sum(dx.ctrl[:]**2)
        # joint_limit_reward = jnp.sum(barrier_cost_quadratic(dx.qpos[7:], lower=-0.5, upper=0.5, margin=0.1, weight=100.0))
        return 10.*height_reward + 0.*rot_ang_reward + 2.*vel_reward + 0.*ctrl_reward 

    ctx = Context(
        lr=3.e-4,
        num_gpu=1,
        seed=10,
        nsteps=200, # 5* (3*ctx.mx.timestep)
        ntotal=200,
        epochs=1000,
        batch=80,
        samples=1,
        eval=5,
        ctrl_dim=12,
        mx=mjx.put_model(model),
        gen_model=lambda: mujoco.MjModel.from_xml_path(model_path),
        gen_network=gen_network,
        run_cost=running_cost,
        terminal_cost=terminal_cost,
        set_data=set_data,
        set_control=set_control,
        controller=policy,
        is_terminal=lambda m, d: jnp.array([False]),
    )

    # from diff_sim.utils.check_init import  check_init_data
    # check_init_data(ctx)

    optimiser = optax.adamw(ctx.lr)
    simulate_fn = eqx.filter_jit(make_simulate_fn_fd(ctx))
    run(ctx, optimiser, simulate_fn, loss_fn_policy_stoch)
