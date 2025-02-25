import os
import time
import jax
jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')
import jax.numpy as jnp
import equinox as eqx
import mujoco
from mujoco import mjx
import optax

import diff_sim
from diff_sim.loss_funcs import loss_fn_policy_det, loss_fn_fitted_policy, loss_fn_fitted_value
from diff_sim.simulation.simulate import make_simulate_fn_fd, make_simulate_fn_mppi
from diff_sim.context.meta_context import Context
from diff_sim.runner_fn import run
from diff_sim.solver.mppi import ParamtersMPPI, mppi
from diff_sim.train_step import step_single_gpu, step_multi_gpu, step_mppi

if __name__ == "__main__":
    # Load mj and mjx model
    model_path = os.path.join(os.path.dirname(diff_sim.__file__), "xmls", "finger_mjx.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    mx = mjx.put_model(model)

    class MLP(eqx.Module):
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
        # Solution of IK
        _, key = jax.random.split(key)
        sign = 2. * jax.random.bernoulli(key, 0.5) - 1.

        # Reference target position in spinner local frame R_s
        r_l = 0.22
        _, key = jax.random.split(key, num=2)
        theta_l = jax.random.uniform(key, (1,), minval=0.6, maxval=2.5)  # Polar Coord
        x_s, y_s = r_l * jnp.cos(theta_l), r_l * jnp.sin(theta_l)  # Cartesian in R_l

        # Reference target position in finger frame R_f
        # Inverse kinematic formula
        x, y = x_s, y_s - 0.39
        l1, l2 = 0.17, 0.161
        q1 = sign * jnp.arccos((x ** 2 + y ** 2 - l1 ** 2 - l2 ** 2) / (2 * l1 * l2))
        q0 = jnp.arctan2(y, x) - jnp.arctan2(l2 * jnp.sin(q1), l1 + l2 * jnp.cos(q1))
        _, key = jax.random.split(key, num=2)
        theta = jax.random.uniform(key, (1,), minval=-0.9, maxval=0.9)

        dx = dx.replace(qpos=dx.qpos.at[0].set(q0[0]))
        dx = dx.replace(qpos=dx.qpos.at[1].set(q1[0]))
        dx = dx.replace(qpos=dx.qpos.at[2].set(theta[0]))
        return dx

    def set_control(dx, u):
        dx = dx.replace(ctrl=dx.ctrl.at[:].set(u))
        return dx

    def gen_network(n: int) -> tuple:
        pkey, vkey = jax.random.split(jax.random.PRNGKey(n))
        return (MLP([6, 128, 256, 128, 2], pkey), MLP([6, 128, 256, 128, 1], vkey))

    def running_cost(mx: mjx.Model, dx: mjx.Data):
        pos_finger = dx.qpos[2]
        u = dx.ctrl
        return 0.002 * jnp.sum(u ** 2) + 0.001 * pos_finger **2

    def terminal_cost(mx: mjx.Model, dx: mjx.Data):
        pos_finger = dx.qpos[2]
        return 4. * pos_finger**2

    # TODO: Add terminal time to is_terminal function
    # Is terminal function only for state, not time condition,
    # handled internally in time.
    def is_terminal(mx: mjx.Model, dx: mjx.Data):
        return jnp.array([False])

    params_mmpi = ParamtersMPPI(
        temp=0.1,
        horizon=8,
        nrollout=10,
        mx=mjx.put_model(model),
        run_cost=running_cost,
        terminal_cost=terminal_cost,
        set_control=set_control
    )

    def policy(nets: eqx.Module, mx: mjx.Model, dx: mjx.Data, policy_key: jnp.ndarray
               ) -> tuple[mjx.Data, jnp.ndarray]:
        # x = jnp.concatenate([dx.qpos, dx.qvel])
        # u = net(x, policy_key)
        net_p, net_v = nets
        def net_p_fn(dx,key):
            x = jnp.concatenate([dx.qpos, dx.qvel])
            return net_p(x, key)
        def net_v_fn(dx,key):
            x = jnp.concatenate([dx.qpos, dx.qvel])
            return net_v(x, key)
        u = mppi(dx,policy_key,net_p_fn, net_v_fn, params_mmpi)
        return dx, u


    ctx = Context(
        lr=3e-3,
        num_gpu=1,
        seed=0,
        nsteps=200,
        ntotal=200,
        epochs=100,
        batch=50,
        samples=1,
        eval=10,
        ctrl_dim=2,
        mx=mjx.put_model(model),
        gen_model=lambda: mujoco.MjModel.from_xml_path(model_path),
        gen_network=gen_network,
        run_cost=running_cost,
        terminal_cost=terminal_cost,
        set_data=set_data,
        set_control=set_control,
        controller=policy,
        is_terminal=is_terminal,
    )

    optimiser = optax.adamw(ctx.lr)
    simulate_fn = eqx.filter_jit(make_simulate_fn_mppi(ctx))
    run(ctx, optimiser, simulate_fn, (loss_fn_fitted_policy, loss_fn_fitted_value), step_mppi)
