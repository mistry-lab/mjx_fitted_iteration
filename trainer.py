import jax
import jax.numpy as jnp
import equinox as eqx
from simulate import controlled_simulate

# @eqx.filter_jit
def loss_fn(params, static, x_init, ctx):
    model = eqx.combine(params, static)
    _,_,costs,_ = controlled_simulate(x_init, ctx, model)
    costs = jnp.sum(costs, axis=1)
    return jnp.mean(costs)

@eqx.filter_jit
def make_step(optim, model, state, loss, x_init, mjmodel, ctx):
    params, static = eqx.partition(model, eqx.is_array)
    loss_value, grads = jax.value_and_grad(loss)(params, static, x_init, mjmodel, ctx)
    updates, state = optim.update(grads, state, model)
    model = eqx.apply_updates(model, updates)
    return model, state, loss_value


@eqx.filter_jit
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
