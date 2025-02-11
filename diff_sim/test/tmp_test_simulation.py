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
from diff_sim.loss_funcs import loss_fn_policy_det
import optax

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
        run_cost=lambda m, d: jnp.sum(5.),
        terminal_cost=lambda m, d: jnp.sum(d.ctrl**2),
        control_cost=lambda m, d: jnp.sum(d.ctrl**2),
        set_data=lambda m, d, x: d,
        gen_network=gen_network,
        is_terminal=lambda m, d: jnp.array([False]),
        set_control=lambda d, u: d,
        controller=lambda net, m, d, k: (d, jnp.zeros((2, 1))),
        loss_func=loss_fn_policy_det
    )
    
    net, optim = ctx.gen_network(ctx.seed), optax.adamw(ctx.lr)

    N = 2000
    keys = jax.vmap(lambda x: jax.random.PRNGKey(0))(jnp.arange(N))
    dxs = jax.vmap(lambda x: mjx.make_data(mx), in_axes=(0,))(jnp.arange(N))
    simulate_fn = make_simulate_fn_fd(ctx, net,100)

    for _ in range(100):
        opt_state = optim.init(eqx.filter(net, eqx.is_array))
        t0 = time.perf_counter_ns()
        model, state, loss_value, res = net.make_step(dxs, optim, net, opt_state, ctx, keys, simulate_fn)
        t1 = time.perf_counter_ns()
        print("Time [ms] : ", 1e-6*(t1 - t0))