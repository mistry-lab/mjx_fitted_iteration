import mujoco
from mujoco import mjx
import jax
import time

is_gpu = next((d for d in jax.devices() if 'gpu' in d.device_kind.lower()), None)
model = mujoco.MjModel.from_xml_path('/home/daniel/Repos/OptimisationBasedControl/models/cartpole.xml')
mx = mjx.put_model(model, device=is_gpu)
dx = mjx.make_data(model)

def get_act_idx(m): return m.actuator_trnid[:, 0]
def get_ctrl_auth(m, dx, idx): return mjx.full_m(m, dx)[:, idx]

def f(m, dx, u, x):
    qpos = dx.qpos.at[:].set(x[0, :])
    qvel = dx.qvel.at[:].set(x[1, :])
    dx = dx.replace(qpos=qpos, qvel=qvel)
    u = dx.ctrl.at[:].set(u)
    dx = dx.replace(ctrl=u)
    mjx.step(m, dx)
    return dx.qacc

@jax.jit
def dfdu_exact(m, dx, x):
    qpos = dx.qpos.at[:].set(x[0, :])
    qvel = dx.qvel.at[:].set(x[1, :])
    dx = dx.replace(qpos=qpos, qvel=qvel)
    return get_ctrl_auth(m, dx, get_act_idx(m))

dfdu_ad = jax.jit(jax.jacobian(f, argnums=2))
u = jax.random.normal(jax.random.PRNGKey(0), shape=(model.nu,))
dx1, dx2 = mjx.step(mx, mjx.make_data(mx)), mjx.step(mx, mjx.make_data(mx))

for _ in range(5):
    start = time.time()
    x = jax.random.normal(jax.random.PRNGKey(0), shape=(2, model.nq)) * 0.3
    dfdu_exact(mx, dx2, x)
    print(f"Exact df/du time {time.time() - start}")

for _ in range(5):
    start = time.time()
    x = jax.random.normal(jax.random.PRNGKey(0), shape=(2, model.nq)) * 0.3
    dfdu_ad(mx, dx1, u, x)
    print(f"AD df/du time {time.time() - start}")
