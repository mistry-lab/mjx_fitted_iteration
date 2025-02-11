import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from diff_sim.simulation.step import make_step_fn_fd, make_step_fn
from diff_sim.simulation.simulate import make_simulate_fn_fd, make_simulate_fn
from diff_sim.context.meta_context import Context
import equinox as eqx
import time
import numpy as np
from diff_sim.nn.base_nn import Network
import os
os.environ["JAX_CHECK_TRACER_LEAKS"] = "True"
jax.checking_leaks()

jax.config.update("jax_enable_x64", True)
jax.config.update('jax_default_matmul_precision', 'high')

if __name__ == "__main__":
    # Load mj and mjx model
    model = mujoco.MjModel.from_xml_path("../xmls/fingers_ball.xml")
    mx = mjx.put_model(model)
    # dxs = jax.vmap(lambda x: mjx.make_data(x[0]), in_axes=(None,0))(mx, jnp.arange(2))
    dxs = jax.vmap(lambda x: mjx.make_data(mx), in_axes=(0,))(jnp.arange(2))


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
            for layer in self.layers[:-1]:
                x = self.act(layer(x))
            x = self.layers[-1](x).squeeze()
            x = jnp.tanh(x) * 1.
            return x
        
    def set_control(dx, u):
        dx = dx.replace(ctrl=dx.ctrl.at[:].set(u))
        return dx

    def gen_network(n: int) -> Network:
        key = jax.random.PRNGKey(n)
        return Policy([6, 128, 128, 2], key)

    ctx = Context(
        lr=4e-3,
        num_gpu=1,
        seed=0,
        nsteps=24,
        ntotal=800,
        epochs=1000,
        batch=2,
        samples=1,
        eval=10,
        ctrl_dim=1,
        mx=mjx.put_model(model),
        gen_model=lambda: mujoco.MjModel.from_xml_string("../xmls/fingers_ball.xml"),
        run_cost=lambda m, d: jnp.sum(d.ctrl**2),
        terminal_cost=lambda m, d: jnp.sum(d.ctrl**2),
        control_cost=lambda m, d: jnp.sum(d.ctrl**2),
        set_data=lambda m, d, x: d,
        gen_network=gen_network,
        is_terminal=lambda m, d: jnp.array([False]),
        set_control=lambda d, u: d,
        controller=lambda net, m, d, k: (d, jnp.zeros((2, 1))),
        loss_func=lambda p, s, d, c, k: (jnp.sum(d.ctrl**2), (jnp.sum(d.ctrl**2), d, jnp.zeros((2, 1)), jnp.zeros((2, 1))))
    )
    net = ctx.gen_network(0)

    # fct_fd = make_step_fn_fd(mx, ctx=ctx)
    N = 20
    keys = jax.vmap(lambda x: jax.random.PRNGKey(0))(jnp.arange(20))
    dxs = jax.vmap(lambda x: mjx.make_data(mx), in_axes=(0,))(jnp.arange(20))
    simulate_fn = make_simulate_fn(ctx, net,20)

    # fct = make_step_fn(mx, ctx)

    # Compute time for call to fct in a loop:
    counter = []
    for _ in range(100):
        
        t0 = time.perf_counter_ns()
        l, x, u, costs, t, terminated = simulate_fn(dxs, keys)
        t1 = time.perf_counter_ns()
        counter.append(t1-t0)
        print("Time [ms] : ", 1e-6*(t1 - t0))
    

    # fnc = make_step_fn(mx, set_control)