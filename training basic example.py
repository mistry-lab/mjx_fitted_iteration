import jax
import jax.numpy as jnp
from jax import random, grad, jit
import optax  # For optimization
import equinox as eqx
from trainer import make_step


class ValueFunc(eqx.Module):
    layers: list
    act: callable

    def __init__(self, dims: list, key):
        keys = jax.random.split(key, len(dims))
        self.layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i], use_bias=True) for i in range(len(dims) - 1)]
        self.act = jax.nn.relu

    def __call__(self, x):
        # transformed_x = self.layers2[0](x)
        # return transformed_x @ x
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)


# System dynamics parameters
A = jnp.array([[1.0]])  # State transition matrix
B = jnp.array([[1.0]])  # Control input matrix

# Cost matrices
Q = jnp.array([[1.0]])  # State cost
R = jnp.array([[0.1]])  # Control cost

# Policy parameter: K (gain matrix)
key = random.PRNGKey(0)
K_init = random.normal(key, (1, 1))

net = ValueFunc([1, 64, 64, 1], jax.random.PRNGKey(0))


def policy(K, x):
    """
    Linear policy: u = -Kx
    """
    return -jnp.dot(K, x)


def dynamics(x, u):
    """
    Linear dynamics: x_next = A x + B u
    """
    return jnp.dot(A, x) + jnp.dot(B, u)


def cost_fn(x, u):
    """
    Quadratic cost: x^T Q x + u^T R u
    """
    return jnp.dot(x.T, jnp.dot(Q, x)) + jnp.dot(u.T, jnp.dot(R, u))


def simulate_trajectory(net, x0, horizon):

    def step(carry, _):
        x = carry
        u = net(x)
        cost = cost_fn(x, u)
        x_next = dynamics(x, u)
        return x_next, (x, cost)

    x_next, (x, cost) = jax.lax.scan(step, x0, None, length=horizon)
    total_cost = jnp.sum(cost)
    return total_cost

def simulate_trajectory2(net, x0, horizon):

    def step(carry, _):
        x = carry
        u = net(x)
        cost = cost_fn(x, u)
        x_next = dynamics(x, u)
        return x_next, (x, cost)

    x_next, (x, cost) = jax.lax.scan(step, x0, None, length=horizon)
    total_cost = jnp.sum(cost)
    return x


# Corrected vmap: Only map over x0_batch (axis=0), keep K and horizon constant
batch_simulate = eqx.filter_jit(jax.vmap(simulate_trajectory, in_axes=(None, 0, None)))
batch_simulate2 = eqx.filter_jit(jax.vmap(simulate_trajectory2, in_axes=(None, 0, None)))


def loss_fn(K, x0_batch, horizon):
    """
    Compute the average total cost over the batch.
    """
    total_costs, _ = batch_simulate(K, x0_batch, horizon)
    return jnp.mean(total_costs)


# @jax.jit
def loss_fn2(params, static, x0_batch, horizon):
    """
    Compute the average total cost over the batch.
    """
    model = eqx.combine(params, static)
    total_costs = batch_simulate(model, x0_batch, horizon)
    return jnp.mean(total_costs)


# Optimization parameters
learning_rate = 0.004
optimizer = optax.adam(learning_rate)
opt_state = optimizer.init(eqx.filter(net, eqx.is_array))

loss_and_grad_fn2 = jit(
    lambda params, static, x0, horizon: (
    loss_fn2(params, static, x0, horizon), grad(loss_fn2)(params, static, x0, horizon)),
    static_argnums=(3,)
)

# Training parameters
num_epochs = 2000
horizon = 50  # Make sure this is a Python integer
batch_size = 100


# @jax.jit
def make_step(optim, model, state, loss, x):
    params, static = eqx.partition(model, eqx.is_array)
    loss_value, grads = jax.value_and_grad(loss)(params, static, x, horizon)
    updates, state = optim.update(grads, state)
    model = eqx.apply_updates(model, updates)
    return model, state, loss_value


for epoch in range(num_epochs):
    # Sample a batch of initial states, e.g., x0 = 0
    x0_batch = jax.random.uniform(key, (batch_size, 1), minval=-1., maxval=1.)  # Starting at 0 for simplicity
    net, opt_state, loss_value = make_step(optimizer, net, opt_state, loss_fn2, x0_batch)

    if epoch % 10 == 0:
        print(f"Epoch {epoch}, Loss: {loss_value}")

# plot traj using batch_simulate
x0 = jnp.array([[1.0]])
traj = batch_simulate2(net, x0, horizon)
# plot it using matplotlib
import matplotlib.pyplot as plt
plt.plot(traj[0, :, 0])
plt.show()




