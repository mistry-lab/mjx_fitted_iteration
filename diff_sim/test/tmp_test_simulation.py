import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from diff_sim.simulation.simulate import make_simulate_fn_fd, make_simulate_fn
from diff_sim.nn.base_nn import step_single_gpu, step_multi_gpu
from diff_sim.context.meta_context import Context
import equinox as eqx
import time
from diff_sim.nn.base_nn import Network
from diff_sim.loss_funcs import loss_fn_policy_det
import optax
from diff_sim.runner_fn import run
import os
import diff_sim
from diff_sim.utils.mj_data_manager import create_data_manager, create

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')

if __name__ == "__main__":
    # Load mj and mjx model
    model_path = os.path.join(os.path.dirname(diff_sim.__file__), "xmls", "fingers_ball.xml")
    model = mujoco.MjModel.from_xml_path(model_path)
    mx = mjx.put_model(model)
    # dxs = jax.vmap(lambda x: mjx.make_data(x[0]), in_axes=(None,0))(mx, jnp.arange(2))
    dxs = jax.vmap(lambda x: mjx.make_data(mx), in_axes=(0,))(jnp.arange(2))

    # @eqx.filter_jit
    class Policy(Network):
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
                # x = self.act( self.dropout(x, key=key))
                x = self.act(x)
            x = self.layers[-1](x).squeeze()
            x = jnp.tanh(x) * 1.
            return x
        
    def set_control(dx, u):
        dx = dx.replace(ctrl=dx.ctrl.at[:].set(u))
        return dx

    def gen_network(n: int) -> Network:
        key = jax.random.PRNGKey(n)
        return Policy([15, 128, 128, 4], key)
    
    def policy(net: Network, mx: mjx.Model, dx: mjx.Data, policy_key: jnp.ndarray
    ) -> tuple[mjx.Data, jnp.ndarray]:
        x = jnp.concatenate([dx.qpos, dx.qvel])
        # t = jnp.expand_dims(dx.time, axis=0)
        # u = net(x)
        # u = 
        # u = 0.5*net(x, t)
        # u += 0.002*jax.random.normal(policy_key, u.shape)
        # Setup offset
        # dx = dx.replace(ctrl=dx.ctrl.at[:].set(u))
        u = jax.random.normal(policy_key,4 ) + net(x, policy_key)

        return dx, u

    ctx = Context(
        lr=4e-3,
        num_gpu=1,
        seed=0,
        nsteps=100,
        ntotal=800,
        epochs=1000,
        batch=200,
        samples=1,
        eval=10,
        ctrl_dim=4,
        mx=mjx.put_model(model),
        gen_model=lambda: mujoco.MjModel.from_xml_path(model_path),
        run_cost=lambda m, d: jnp.sum(5.),
        terminal_cost=lambda m, d: jnp.sum(d.ctrl**2),
        control_cost=lambda m, d: jnp.sum(d.ctrl**2),
        set_data=lambda m, d, k: d,
        gen_network=gen_network,
        is_terminal=lambda m, d: jnp.array([False]),
        set_control=lambda d, u: d,
        controller=policy,
        loss_func=loss_fn_policy_det
    )
    
    net, optim = ctx.gen_network(ctx.seed), optax.adamw(ctx.lr)
    params, static = eqx.partition(net, eqx.is_array)

    N = ctx.batch
    keys = jax.vmap(lambda x: jax.random.PRNGKey(0))(jnp.arange(N))
    simulate_fn = make_simulate_fn_fd(ctx)

    data_manager = create_data_manager()
    dxs = create(mx, ctx.batch)
   
    for _ in range(100):
        # dxs = create(mx, 2000)
        opt_state = optim.init(eqx.filter(net, eqx.is_array))
        t0 = time.perf_counter_ns()
        model, state, loss_value, res = step_single_gpu(dxs, optim, net, opt_state, ctx, keys, simulate_fn)
        t1 = time.perf_counter_ns()
        print("Time [ms] : ", 1e-6*(t1 - t0))

    # run(ctx, optim, simulate_fn)

    # runner = Runner()
    # runner.run(ctx, simulate_fn, optimiser)
    # run(ctx, optimiser, simulate_fn)