import jax
from jax import numpy as jnp
import equinox as eqx
from .meta_context import Config, Callbacks, Context
import os
import jax.debug
from mujoco import mjx
import mujoco

try:
    # This works when __file__ is defined (e.g., in regular scripts)
    base_path = os.path.dirname(__file__)
except NameError:
    # Fallback to current working directory (e.g., in interactive sessions)
    base_path = os.getcwd()

base_path = os.path.join(base_path, '../xmls/unitree_a1')

def set_simulator_options(mj_model):
    # Set integrator (Euler or RK4)
    mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_EULER   # Use Euler integrator
    # mj_model.opt.integrator = mujoco.mjtIntegrator.mjINT_RK4   # Use RK4 if desired

    # Set friction cone mj_model (Elliptic or Pyramidal)
    mj_model.opt.cone = mujoco.mjtCone.mjCONE_ELLIPTIC   # Elliptic friction cone
    # mj_model.opt.cone = mujoco.mjtCone.mjCONE_PYRAMIDAL # Pyramidal friction cone

    # Set Jacobian calculation (Automatic)
    mj_model.opt.jacobian = mujoco.mjtJacobian.mjJAC_AUTO  # Auto-select Jacobian

    # Set the solver to Newton
    # mj_model.opt.solver = mujoco.mjtSolver.mjSOL_NEWTON   # Newton solver
    # Alternatives:
    mj_model.opt.solver = mujoco.mjtSolver.mjSOL_CG    # Conjugate Gradient solver
    # mj_model.opt.solver = mujoco.mjtSolver.mjSOL_PGS   # Projected Gauss-Seidel solver

    # Set solver iterations and tolerance
    mj_model.opt.iterations = 1         # Solver iterations
    mj_model.opt.tolerance = 1e-8        # Solver tolerance

    # Set no-slip tolerance and iterations
    mj_model.opt.noslip_tolerance = 1e-6   # No-slip tolerance
    mj_model.opt.noslip_iterations = 0     # No-slip iterations (0 or 3 if needed)

    # Set MPR tolerance
    mj_model.opt.mpr_tolerance = 1e-6      # MPR tolerance

    # Enable contact override
    # mj_model.opt.enableflags = mujoco.mjtEnableBit.mjENBL_OVERRIDE

    # Set solver impedance (solimp), margin, and solver reference (solref)
    # mj_model.opt.solimp[0] = 0.28   # Solver impedance solimp[0]
    # mj_model.opt.solimp[1] = 0.65   # Solver impedance solimp[1]
    # mj_model.opt.solimp[2] = 0.02   # Solver impedance solimp[2]

    # mj_model.opt.margin = 0.001     # Contact margin

    # mj_model.opt.solref[0] = 0.005  # Solver reference solref[0]

    # Disable features: joint limits, equality constraints, filtering parent, midphase
    mj_model.opt.disableflags = (mujoco.mjtDisableBit.mjDSBL_LIMIT | 
                            mujoco.mjtDisableBit.mjDSBL_EQUALITY | 
                            mujoco.mjtDisableBit.mjDSBL_FILTERPARENT | 
                            mujoco.mjtDisableBit.mjDSBL_MIDPHASE)
    return None

class ValueFunc(eqx.Module):
    layers: list
    act: callable

    def __init__(self, dims: list, key):
        keys = jax.random.split(key, len(dims))
        self.layers = [eqx.nn.Linear(dims[i], dims[i + 1], key=keys[i], use_bias=True) for i in range(len(dims) - 1)]
        self.act = jax.nn.relu     

    def __call__(self, x, t):
        t = t if t.ndim == 1 else t.reshape(1)
        x = jnp.concatenate([x, t], axis=-1)
        for layer in self.layers[:-1]:
            x = self.act(layer(x))
        return self.layers[-1](x)
    
    @staticmethod
    def make_step(optim, model, state, loss, x, y):
        params, static = eqx.partition(model, eqx.is_array)
        loss_value, grads = jax.value_and_grad(loss)(params, static, x, y)
        updates, state = optim.update(grads, state, model)
        model = eqx.apply_updates(model, updates)
        return model, state, loss_value

    @staticmethod
    def loss_fn(params, static, x, y):
        model = eqx.combine(params, static)
        pred = jax.vmap(model)(x.reshape(-1, x.shape[-1]))
        y = y.reshape(-1, 1)
        return jnp.mean(jnp.square(pred - y))


def policy(x,t,net,cfg, mx, dx):
    return net(x,t)

def hjb(x,t,net,cfg, mx, dx):
    Kp = jnp.diag(jnp.concatenate([jnp.zeros(6),mx.actuator_gainprm[:,0]])) 
    # act_id = mx.actuator_trnid[:, 0]
    M = mjx.full_m(mx, dx) 
    invM_Kp = jnp.linalg.inv(M) @ Kp 
    dvdx = jax.jacrev(net,0)(state_encoder(x, mx), t)
    G = jnp.vstack([jnp.zeros_like(invM_Kp), invM_Kp])
    invR = jnp.linalg.inv(cfg.R)
    u = (-1/2 * invR @ G.T @ dvdx.T).flatten()
    # jax.debug.print("u : {uu}", uu=u)
    return u[-mx.nu:]    

def get_init(batch, key):
    pose = jnp.array([0.,0.,0.25,1.,0.,0.,0.])
    # q_init = jax.random.uniform(key, (batch, 1), minval=-1., maxval=1.)
    q_init = jnp.zeros(12)
    vel = jnp.array([0.,0.,0.,0.,0.,0.])
    qvel = jnp.zeros(12)
    return jnp.concatenate([
            jnp.tile(pose, (batch, 1)),
            jnp.tile(q_init, (batch, 1)),
            jnp.tile(vel, (batch, 1)),
            jnp.tile(qvel, (batch, 1))
        ], axis=1)


def state_encoder(x, mx):
    p, v = x[:mx.nq], x[mx.nq:]
    mrp = p[1:4]/(p[0] + 1)
    return jnp.concatenate([mrp, p[4:], v])


def control_cost(u):
    return jnp.dot(u.T, jnp.dot(jnp.diag(100.*jnp.ones([12])), u))


ctx = Context(cfg=Config(
    model_path=os.path.join(base_path, 'task_hill.xml'),
    dims=[2, 64, 64, 1],
    lr=4.e-3,
    seed=0,
    nsteps=200,
    epochs=400,
    batch=4,
    vis=10,
    dt=0.01,
    R=jnp.diag(100.*jnp.ones(18)),
    horizon=jnp.arange(0, (200 )*0.01, 0.01) + 0.01
    ),cbs=Callbacks(
        # run_cost= lambda x: jnp.einsum('...ti,ij,...tj->...t', x, jnp.diag(jnp.array([10., .1])), x),
        # terminal_cost= lambda x: jnp.einsum('...ti,ij,...tj->...t', x, jnp.diag(jnp.array([1., .01])), x),
        # terminal_cost = lambda x: 10*jnp.sum(jnp.abs(jnp.dot(jnp.diag(jnp.array([10., 0.1])), x.T).T), axis=-1),
        # control_cost= lambda x: jnp.einsum('...ti,ij,...tj->...t', x, jnp.array([[.1]]), x).at[..., -1].set(0),
        run_cost= lambda x: jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([10.,0.1])), x)),
        control_cost= control_cost,
        terminal_cost= lambda x: 10*jnp.dot(x.T, jnp.dot(jnp.diag(jnp.array([10.,0.1])), x)),
        init_gen= get_init,
    state_encoder=state_encoder,
    gen_network = lambda : ValueFunc([37, 64,128,64, 1], jax.random.PRNGKey(0)),
    controller = hjb
    )
)

