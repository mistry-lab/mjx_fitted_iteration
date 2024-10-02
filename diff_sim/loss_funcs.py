import jax
import jax.numpy as jnp
import equinox as eqx
from jaxtyping import PyTree
from diff_sim.context.meta_context import Context
from simulate import controlled_simulate

def loss_fn(params: PyTree, static: PyTree, x_init: jnp.ndarray, ctx: Context) -> jnp.ndarray:
    """
        Loss function for the policy optimization problem
        Args:
            params: PyTree, model parameters
            static: PyTree, static parameters
            x_init: jnp.ndarray, initial state
            ctx: Context, context object
        Returns:
            jnp.ndarray, loss value

        Mathematical Formulation:
        We compute the sum of the costs over the entire trajectory and average it over the batch
        loss = 1/B * sum_{b=1}^{B} sum_{t=1}^{T} cost(x_{b,t}, u_{b,t})
    """
    model = eqx.combine(params, static)
    _,_,costs,_ = controlled_simulate(x_init, ctx, model)
    costs = jnp.sum(costs, axis=1)
    return jnp.mean(costs)


# @eqx.filter_jit
def loss_fn_td(params, static, x_init, mjmodel, ctx):
    @jax.vmap
    def v_diff(x,t):
        v_seq = jax.vmap(model)(x, t)
        v0, v1 = v_seq[0:-1], v_seq[1:]
        return v0 - v1, v_seq[-1]

    @jax.vmap
    def v_r_cost(diff, term, cost):
        v_diff_cost = diff - cost[:-1]
        v_term_cost = term - cost[-1]
        return jnp.sum(jnp.square(v_diff_cost) + jnp.square(v_term_cost))
    
    model = eqx.combine(params, static)
    x,_,costs,t = controlled_simulate(x_init, mjmodel, ctx, model)
    B, T, _ = x.shape
    diff, term = v_diff(x,t)
    costs = v_r_cost(diff.reshape(B, T-1), term.reshape(B, 1), costs)
    return jnp.mean(costs)

def loss_fn_target(params, static, x_init, mjmodel, ctx):

    @jax.vmap
    def cost(x, costs):
        pred = jax.vmap(model)(x, ctx.cfg.horizon)
        targets = jnp.flip(costs)
        targets = jnp.flip(jnp.cumsum(targets))
        return jnp.sum(jnp.square(pred - targets))

    model = eqx.combine(params, static)
    x,_,costs =  controlled_simulate(x_init, mjmodel, ctx, model, PD=True)
    return jnp.mean(cost(x,costs))
