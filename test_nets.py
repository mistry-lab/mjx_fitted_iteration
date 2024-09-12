import jax
import wandb
import equinox as eqx
import optax
import jax.numpy as jnp
import jax.debug
import matplotlib.pyplot as plt
import flax.nnx as nnx
import flax

from clu import metrics
from flax.training import train_state  # Useful dataclass to keep train state
from flax import struct                # Flax dataclasses
from flax.linen import initializers

###################################
# Value Function Equinox
###################################

# Eqx module
class ValueFunc(eqx.Module):
    layers: list
    act: callable

    def __init__(self, dims: list, key):
        keys = jax.random.split(key, len(dims))
        self.layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i], use_bias=True) for i in range(len(dims) - 1)]
        self.act = jax.nn.relu

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)
    
def make_step(optim, model, state, loss, x, y):
    params, static = eqx.partition(model, eqx.is_array)
    loss_value, grads = jax.value_and_grad(loss)(params, static, x, y)
    updates, state = optim.update(grads, state, model)
    model = eqx.apply_updates(model, updates)
    return model, state, loss_value

def loss_fn_eqx(params, static, x, y):
    model = eqx.combine(params, static)
    pred = jax.vmap(model)(x.reshape(-1, x.shape[-1]))
    y = y.reshape(-1, 1)
    return jnp.mean(jnp.square(pred - y))

###################################
# Value Function Flax
###################################

class ValueFuncFLAX(flax.linen.Module):
    @flax.linen.compact
    def __call__(self, x):
        x = flax.linen.Dense(64,kernel_init=initializers.variance_scaling(0.57, "fan_in", "uniform"), bias_init=initializers.zeros_init())(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(64,kernel_init=initializers.variance_scaling(0.57, "fan_in", "uniform"), bias_init=initializers.zeros_init())(x)
        x = flax.linen.relu(x)
        x = flax.linen.Dense(1,kernel_init=initializers.variance_scaling(0.57, "fan_in", "uniform"), bias_init=initializers.zeros_init())(x)
        return x

@struct.dataclass
class Metrics(metrics.Collection):
  accuracy: metrics.Accuracy
  loss: metrics.Average.from_output('loss')

class TrainState(train_state.TrainState):
  metrics: Metrics

def create_train_state(module, rng, learning_rate):
    """Creates an initial `TrainState`.
    """
    batch = jnp.ones((10000,2))
    params = module.init(rng, batch)['params'] # initialize parameters by passing a template image
    #   tx = optax.sgd(learning_rate, momentum)
    tx = optax.adamw(learning_rate)
    return TrainState.create(
        apply_fn=module.apply, params=params, tx=tx,
        metrics=Metrics.empty())

@jax.jit
def train_step_flax(state, batch, target):
  """Train for a single step."""
  def loss_fn(params):
    values = state.apply_fn({'params': params}, batch)
    loss = jnp.mean(jnp.square(values - target))
    return loss
  grad_fn = jax.value_and_grad(loss_fn)
  loss, grads = grad_fn(state.params)
  state = state.apply_gradients(grads=grads)
  return state, loss

############################
# Valude Function NNX
############################

class ValueFuncNNX(nnx.Module):

    def __init__(self, dims: list, rngs: nnx.Rngs):
        self.layers = [nnx.Linear(dims[i], dims[i + 1], use_bias=True, rngs=rngs) for i in range(len(dims) - 1)]
        self.act = nnx.relu

    def __call__(self, x):
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)



#############################################################################################################

wandb.init(project="fvi", anonymous="allow", mode='online')

# Function to fit
def sincos_2d(x):
    return jnp.sin(4*x[0]) + jnp.cos(4*x[1])
    # return x[0]**2 + x[1]**2


def loss_fn(model: ValueFuncNNX, x, y):
    y_pred = model(x)
    return jnp.mean(jnp.square(y_pred - y))

@nnx.jit  # automatic state management
def train_step(model: ValueFuncNNX, optimizer:nnx.Optimizer, x, y):

    loss, grads = nnx.value_and_grad(loss_fn)(model, x, y)
    optimizer.update(grads)  # inplace updates

    return loss


if __name__ == '__main__':
    try:
        lr = 4e-3
        # Equinox Optimiser 
        net_eqx = ValueFunc([2, 64,64, 1], jax.random.PRNGKey(0))
        optim = optax.adamw(lr)
        opt_state = optim.init(eqx.filter(net_eqx, eqx.is_array))

        # NNX Optimiser
        net_nnx =ValueFuncNNX([2, 64,64, 1], nnx.Rngs(50))
        optimizer_nnx = nnx.Optimizer(net_nnx, optax.adamw(lr))

        # Flax Optimiser
        net_flax = ValueFuncFLAX()
        state = create_train_state(net_flax, jax.random.key(0), lr)
        # Random number generator
        init_key = jax.random.PRNGKey(0)
        key, subkey = jax.random.split(init_key)

        with jax.default_device(jax.devices()[0]):    
            # Jit equinox modules
            f_make_step = eqx.filter_jit(make_step)
            f_sincos_2d = eqx.filter_jit(jax.vmap(sincos_2d, in_axes=(0)))            
            
            # Generate inputs and targets
            x = jnp.linspace(-1, 1, 100)
            xv, yv = jnp.meshgrid(x, x)
            x_inits = jnp.stack([xv.ravel(), yv.ravel()], axis=1)
            target = f_sincos_2d(x_inits)
            
            N_iteration = 75
            for _ in range(N_iteration):
                net_eqx, opt_state, loss_eqx = f_make_step(optim, net_eqx,opt_state, loss_fn_eqx,x_inits, target)
                loss_nnx = train_step(net_nnx, optimizer_nnx,x_inits, target )
                state, loss_flax = train_step_flax(state, x_inits, target)

                # Log the loss
                wandb.log({"loss_eqx": loss_eqx, "loss_nnx": loss_nnx, "loss_flax":loss_flax})

            # Stack the input data into a batch
            print("\nPlotting ...")
            xy = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)  # (10000, 2)

            # Vectorize the operations using vmap
            net_func_equinox = jax.vmap(lambda xy: net_eqx(xy))
            net_func_nnx = jax.vmap(lambda xy: net_nnx(xy))
            net_func_flax = jax.vmap(lambda xy: state.apply_fn({'params': state.params}, xy))

            # Optionally use jit to compile the functions
            net_func_equinox = jax.jit(net_func_equinox)
            net_func_nnx = jax.jit(net_func_nnx)
            net_func_flax = jax.jit(net_func_flax)

            # Apply the vectorized and jitted functions to the batched input
            z = net_func_equinox(xy).reshape(xv.shape)
            zz = net_func_nnx(xy).reshape(xv.shape)
            zzz = net_func_flax(xy).reshape(xv.shape)
            
            fig, axes = plt.subplots(2, 2, subplot_kw={"projection": "3d"}, figsize=(10, 10))

            # Plotting the reference
            axes[0,0].plot_surface(xv, yv, target.reshape(100,100), cmap='viridis')
            axes[0,0].set_title('Reference')
            axes[0, 0].set_xlabel('X')
            axes[0, 0].set_ylabel('Y')

            # Plotting the first surface (z)
            axes[0,1].plot_surface(xv, yv, z, cmap='viridis')
            axes[0,1].set_title('Net-Equinox')
            axes[0,1].set_xlabel('X')
            axes[0,1].set_ylabel('Y')

            # Plotting the second surface (zz)
            axes[1,0].plot_surface(xv, yv, zz, cmap='viridis')
            axes[1,0].set_title('Net-NNX')
            axes[1,0].set_xlabel('X')
            axes[1,0].set_ylabel('Y')

            # Plotting the third surface (zzz)
            axes[1,1].plot_surface(xv, yv, zzz, cmap='viridis')
            axes[1,1].set_title('Net-FLAX')
            axes[1,1].set_xlabel('X')
            axes[1,1].set_ylabel('Y')

            # Adjusting layoutloss_fn
            fig.suptitle(f"Comparison Equinox, NNX, Flax. Iteration: {N_iteration}, Optimiser: Adamw")
            # plt.tight_layout()
            plt.show()


    except KeyboardInterrupt:
        print("Exit wandb")
        wandb.finish()
