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
from diff_sim.loss_funcs import loss_fn_policy_det
from diff_sim.simulation.simulate import make_simulate_fn_fd, make_simulate_fn
from diff_sim.train_step import step_single_gpu, step_multi_gpu
from diff_sim.context.meta_context import Context
from diff_sim.utils.mj_data_manager import create_data_manager

from diff_sim.runner_fn import run

if __name__ == "__main__":
    # Load mj and mjx model
    model_path = os.path.join(os.path.dirname(diff_sim.__file__), "xmls", "fingers_ball.xml")
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
        theta1 = jax.random.uniform(key, (1,), minval=0.45, maxval=0.7) # proximal1
        theta2 = jnp.array([-0.6]) # distal1
        _, key = jax.random.split(key)
        theta3 = jax.random.uniform(key, (1,), minval=-.7, maxval=-.45) # proximal2
        theta4 = jnp.array([0.6]) # distal2

        # init_quat = jnp.array([1.0, 0.,0.,0.]) # ball
        _, key = jax.random.split(key)
        init_angl = jax.random.uniform(key, (1,), minval=-1.2, maxval=1.2) # proximal2
        qpos = jnp.concatenate([theta1, theta2, theta3, theta4, init_angl])
        qvel = jnp.zeros(mx.nv)
        qvel = qvel.at[0].set(-0.)
        qvel = qvel.at[2].set(0.)

        dx = dx.replace(qpos=dx.qpos.at[:].set(qpos), qvel=dx.qvel.at[:].set(qvel))

        return dx

    def set_control(dx, u):
        dx = dx.replace(ctrl=dx.ctrl.at[:].set(u))
        # dx = dx.replace(qfrc_applied=dx.qfrc_applied.at[6].set(u[0]))
        return dx

    def gen_network(n: int) -> eqx.Module:
        key = jax.random.PRNGKey(n)
        return Policy([10, 128,256,128, 4], key)

    def policy(net: eqx.Module, mx: mjx.Model, dx: mjx.Data, policy_key: jnp.ndarray
    ) -> tuple[mjx.Data, jnp.ndarray]:
        x = jnp.concatenate([dx.qpos, dx.qvel])
        u = net(x, policy_key)

        return dx, u


    def running_cost(mx: mjx.Model, dx: mjx.Data):
        # quat_ref = axis_angle_to_quat(jnp.array([0.,0.,1.]), jnp.array([2.35]))
        # costR = jnp.sum((quat_to_mat(dx.qpos[4:8])  - quat_to_mat(quat_ref))**2)
        c = dx.qpos[4] - 2.35
        return  0.001*c**2 + 0.*jnp.sum(dx.ctrl**2)

    def terminal_cost(mx: mjx.Model, dx: mjx.Data):
        # quat_ref = axis_angle_to_quat(jnp.array([0.,0.,1.]), jnp.array([2.35]))
        # costR = jnp.sum((quat_to_mat(dx.qpos[4:8])  - quat_to_mat(quat_ref))**2)
        c = dx.qpos[4] - 2.35
        return 4.*c**2

    ctx = Context(
        lr=1.e-3,
        num_gpu=1,
        seed=0,
        nsteps=200,
        ntotal=200,
        epochs=1000,
        batch=50,
        samples=1,
        eval=15,
        ctrl_dim=4,
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

    # optimiser = optax.adamw(ctx.lr)
    # opt_state = optim.init(eqx.filter(net, eqx.is_array))
    # params, static = eqx.partition(net, eqx.is_array)
    # key_init = jax.random.PRNGKey(0)
    # key_data, key_sim = jax.random.split(key_init, num=2)
    # simulate_fn = make_simulate_fn_fd(ctx)

    # data_manager = create_data_manager()
    # dxs = data_manager.create_data(ctx, key_data)

    # for e in range(100):
    #     t0 = time.perf_counter_ns()
    #     model, state, loss_value, res = step_single_gpu(
    #         dxs, net, ctx, key_sim, opt_state, optim, simulate_fn, loss_fn_policy_det
    #     )
    #     t1 = time.perf_counter_ns()
    #     print("Time [ms] : ", 1e-6*(t1 - t0), "epoch: ", e)


    optimiser = optax.adamw(ctx.lr)
    simulate_fn = eqx.filter_jit(make_simulate_fn_fd(ctx))
    run(ctx, optimiser, simulate_fn, loss_fn_policy_det)
