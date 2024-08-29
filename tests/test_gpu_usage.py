import sys
import time
import os
import jax
import jax.numpy as jnp
import numpy as np
import mujoco
from mujoco import mjx
sys.path.append(os.path.abspath(os.path.join(os.path.dirname(__file__), '..')))
from utils.mj_vis import animate_trajectory
from utils.tqdm import trange
model = mujoco.MjModel.from_xml_path('../xmls/cartpole.xml')
data = mujoco.MjData(model)
mjx_model = mjx.put_model(model)

# TODO pass in mx as an argument

def simulate_single(x, nsteps):
    def batched_step(mjx_data, _):
        dx = jax.lax.stop_gradient(mjx.step(mjx_model, mjx_data))
        return dx, jnp.concatenate([dx.qpos, dx.qvel], axis=0)

    def scan_fn(dx, _):
        # Perform a batched simulation step over a sequence of steps
        return jax.lax.scan(batched_step, dx, None, length=nsteps)

    def set_init(dx, x):
        qpos = x[:model.nq]  # Extract position part
        qvel = x[model.nq:]  # Extract velocity part
        dx = dx.replace(qpos=qpos, qvel=qvel)
        return dx

    dx = mjx.make_data(mjx_model)
    dx = set_init(dx, x)
    dx_batched = mjx.step(mjx_model, dx)
    _, traj = scan_fn(dx_batched, None)

    return traj

@jax.jit
def simulate(x, nsteps=100):
    trajs = jax.vmap(simulate_single, in_axes=(0, None))(x, nsteps)
    return trajs


stats = []
for e in trange(10):
  start = time.time()
  pos = jax.random.uniform(jax.random.PRNGKey(0), (1000, model.nq), minval=-0.1, maxval=0.1)
  vel = jax.random.uniform(jax.random.PRNGKey(0), (1000, model.nv), minval=-0.1, maxval=0.1)
  x = jnp.concatenate([pos, vel], axis=-1)  # This should give x shape (1000, nq + nv)
  traj = simulate(x)
  stats.append((f"Epoch {e}", time.time() - start))

# Print the stats in a nice table
print("\nSimulation Statistics:")
print("+" + "-"*20 + "+" + "-"*16 + "+")
print(f"| {'Epoch':<18} | {'Time (seconds)':<14} |")
print("+" + "="*20 + "+" + "="*16 + "+")
for epoch, time in stats:
    print(f"| {epoch:<18} | {time:<14.4f} |")
    print("+" + "-"*20 + "+" + "-"*16 + "+")

candidate = np.array(traj)
animate_trajectory(candidate, data, model)
