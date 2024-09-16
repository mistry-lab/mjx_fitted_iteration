from simulate import controlled_simulate
from contexts.di import ctx as di_ctx
import jax
import jax.numpy as jnp
import mujoco
from mujoco import mjx
from trainer import gen_traj_targets, make_step, loss_fn_td, loss_fn_target, gen_traj_cost
import equinox as eqx
import optax
import matplotlib.pyplot as plt
import copy 
import wandb
from utils.mj_vis import animate_trajectory
import numpy as np

wandb.init(project="fvi", anonymous="allow", mode='online')

def ref(x):
    Q = jnp.diag(jnp.array([1,1]))
    return x @ Q @ x.T

# get keys and generate init conditions
key = jax.random.PRNGKey(0)
key, subkey = jax.random.split(key)

# generate a grid of initial conditions using a meshgrid
x = jnp.linspace(-1, 1, 100)
y = jnp.linspace(-0.25, 0.25, 100)
xv, yv = jnp.meshgrid(x, y)
x_inits = jnp.stack([xv.ravel(), yv.ravel()], axis=1)

# rollout the dynamics using these initial conditions with mujoco double integrator
model = mujoco.MjModel.from_xml_path(di_ctx.cfg.model_path)
model.opt.timestep = di_ctx.cfg.dt
data = mujoco.MjData(model)
mx = mjx.put_model(model)

di_ctx2 = copy.deepcopy(di_ctx)

sim = eqx.filter_jit(jax.vmap(controlled_simulate, in_axes=(0, None, None, None)))
f_target = eqx.filter_jit(jax.vmap(gen_traj_targets, in_axes=(0, 0, None)))
f_cost = eqx.filter_jit(jax.vmap(gen_traj_cost, in_axes=(0, 0, None)))
f_make_step = eqx.filter_jit(make_step)

# Optimiser
optim = optax.adamw(di_ctx.cfg.lr)
opt_state = optim.init(eqx.filter(di_ctx.cbs.net, eqx.is_array))

optim2 = optax.adamw(di_ctx.cfg.lr)
opt_state2 = optim2.init(eqx.filter(di_ctx2.cbs.net, eqx.is_array))

x, u = sim(x_inits, mx, di_ctx, PD=False)
x2, u2 = sim(x_inits, mx, di_ctx2, PD=False)

# calculate the target values from out function
target, total_cost, terminal_cost = f_target(x, u, di_ctx)
target2, total_cost2, terminal_cost2 = f_cost(x2, u2, di_ctx2)

times = jnp.repeat(di_ctx.cfg.horizon.reshape(1,di_ctx.cfg.horizon.shape[-1]), 10000, axis=0)

for e in range(100):

    x, u = sim(x_inits, mx, di_ctx, PD=False)
    x2, u2 = sim(x_inits, mx, di_ctx2, PD=False)
    for k in range(5):
        di_ctx.cbs.net, opt_state, loss_value = make_step(optim, di_ctx.cbs.net,opt_state, loss_fn_target, x, times, target)
        di_ctx2.cbs.net, opt_state2, loss_value2 = make_step(optim2, di_ctx2.cbs.net,opt_state2, loss_fn_td, x, times, target2)
        wandb.log({"loss-target": loss_value, "loss-td": loss_value2})

    # if e % di_ctx.cfg.vis == 0 :
    #     key, subkey = jax.random.split(key)
    #     xinits = x_inits[jax.random.randint(key, (10,), 0, 700),:]
    #     # Simultate the 5 runs with the Optimised controller.
    #     xtraj, utraj = sim(xinits, mx, di_ctx, PD=False)
    #     animate_trajectory(xtraj, data, model)

# Visualisation
# xs = x[jax.random.randint(key, (5,), 0, di_ctx.cfg.nsteps)]
# term_cst = di_ctx.cbs.terminal_cost(xs[...,-1,:])
# print(f"Terminal cost is: {term_cst}")

# Sample 5 initial conditions in x_inits
print("10 Sample from loss-target controller")
key, subkey = jax.random.split(key)
xinits = x_inits[jax.random.randint(key, (10,), 0, 700),:]
# Simultate the 5 runs with the Optimised controller.
xtraj, utraj = sim(xinits, mx, di_ctx, PD=False)
animate_trajectory(xtraj, data, model)

print("10 Sample from loss-td controller")
key, subkey = jax.random.split(key)
xinits = x_inits[jax.random.randint(key, (10,), 0, 700),:]
# Simultate the 5 runs with the Optimised controller.
xtraj, utraj = sim(xinits, mx, di_ctx, PD=False)
animate_trajectory(xtraj, data, model)

# f_ref = jax.jit(jax.vmap(ref, in_axes=(0)))    
# costs = f_ref(x_inits)

net_func = jax.jit(jax.vmap(lambda x,t: di_ctx.cbs.net(x,t)))
xy = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)  # (10000, 2)
z = net_func(xy, times[:,0].reshape(-1,1)).reshape(xv.shape)
fig = plt.figure("NET FIT T=0")
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xv, yv, z, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')

net_func2 = jax.jit(jax.vmap(lambda x,t: di_ctx2.cbs.net(x,t)))
xy = jnp.stack([xv.ravel(), yv.ravel()], axis=-1)  # (10000, 2)
zz = net_func2(xy, times[:,0].reshape(-1,1)).reshape(xv.shape)
fig = plt.figure("NET-TD FIT T=0")
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xv, yv, zz, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')


v = terminal_cost.reshape(100, 100)
# v = terminal_cost2.reshape(100,100)
fig = plt.figure("Value at t=0 (total sum of cost)")
ax = fig.add_subplot(111, projection='3d')
ax.plot_surface(xv, yv, v, cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')



# plot the targets as a meshgrid with their associated states the first target is associated with the first state

# fig = plt.figure("Value at t=N, terminal cost")
# vv = terminal_cost.reshape(100, 100)
# ax = fig.add_subplot(111, projection='3d')
# ax.plot_surface(xv, yv, vv, cmap='viridis')
# ax.set_xlabel('X')
# ax.set_ylabel('Y')

fig = plt.figure("Distribution (x, xdot) at Terminal Time")
vvv = jnp.ones(10000)
# x_i = jnp.zeros(10000)
# y_i = jnp.zeros(10000)
# for i,xs in enumerate(x):
#     x_i[i] = xs[0]
#     y_i[i] = xs[1]
x_i = x[:,-1, 0]  # Extract all first elements (first column)
y_i = x[:,-1, 1]  # Extract all second elements (second column)
print("x.shape : ", x.shape)
print("x_i.shape : ", x_i.shape)
print("y_i.shape : ", y_i.shape)
ax = fig.add_subplot(111, projection='3d')
# ax.plot3D(x_i, y_i, vvv, cmap='viridis')
# Use scatter3D for a 3D scatter plot with a colormap
sc = ax.scatter3D(x_i.flatten(), y_i.flatten(), vvv.flatten(), c=vvv.flatten(), cmap='viridis')
ax.set_xlabel('X')
ax.set_ylabel('Y')
# Add a color bar to reflect the color mapping
fig.colorbar(sc)

# plt.show()
plt.show()