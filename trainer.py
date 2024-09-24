import jax
import jax.numpy as jnp
import equinox as eqx
from simulate import controlled_simulate

# @eqx.filter_jit
def loss_fn(params, static, x_init, mjmodel, ctx):
    model = eqx.combine(params, static)
    _,_,costs,_ = controlled_simulate(x_init, mjmodel, ctx, model)
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
    # pred = jax.vmap(model)(x, ctx.cfg.horizon)
    # pred = pred.reshape(y.shape)
    # loss = jnp.sum(jnp.square(pred - y), axis=-1)
    return jnp.mean(cost(x,costs))

# def gen_traj_targets(x, u, ctx:Context):
#     xs, xt = x[:-1], x[-1]
#     xt = xt if len(xt.shape) == 2 else xt.reshape(1, xt.shape[-1])
#     ucost = ctx.cbs.control_cost(u) * ctx.cfg.dt
#     xcst = ctx.cbs.run_cost(xs) * ctx.cfg.dt
#     tcst = ctx.cbs.terminal_cost(xt)
#     xcost = jnp.concatenate([xcst, tcst]) + ucost
#     costs = jnp.flip(xcost)
#     targets = jnp.flip(jnp.cumsum(costs))
#     total_cost = targets[0]    
    
#     return targets, total_cost, tcst

# def gen_traj_cost(x, u, ctx:Context):
#     xs, xt = x[:-1], x[-1]
#     xt = xt if len(xt.shape) == 2 else xt.reshape(1, xt.shape[-1])
#     ucost = ctx.cbs.control_cost(u) * ctx.cfg.dt
#     xcst = ctx.cbs.run_cost(xs) * ctx.cfg.dt
#     tcst = ctx.cbs.terminal_cost(xt)
#     cost = jnp.concatenate([xcst, tcst]) + ucost
#     return cost, jnp.sum(cost), tcst



# def make_step(optim, model, state, loss, x, times, y):
#     params, static = eqx.partition(model, eqx.is_array)
#     # static = jax.lax.stop_gradient(static)

#     # loss_value, grads = eqx.filter_value_and_grad(loss)(model, x, times, y)
#     loss_value, grads = jax.value_and_grad(loss)(params, static, x, times, y)

#     # check_grads(loss, params, order=2)  # check up to 2nd order derivatives

#     jax.debug.print("grads : {gr}", gr=grads)
#     updates, state = optim.update(grads, state, model)
#     model = eqx.apply_updates(model, updates)
#     return model, state, loss_value




# args x_inits, mx, ctx, net, PD=False


# def loss_fn_td(params, static, x, times, cost):
#     model = eqx.combine(params, static)
#     B, T, nx = x.shape

#     def v_diff(x, t):
#         v_seq = jax.vmap(model)(x, t)
#         v0, v1 = v_seq[0:-1], v_seq[1:]
#         return v0 - v1, v_seq[-1]

#     def v_r_cost(v_diff, v_term, cost):
#         v_diff_cost = v_diff - cost[:-1]
#         v_term_cost = v_term - cost[-1]
#         return jnp.sum(jnp.square(v_diff_cost) + jnp.square(v_term_cost))

#     # v_diff, v_term = jax.vmap(v_diff)(x, times)
#     # cost = jax.vmap(v_r_cost)(v_diff.reshape(B, T-1), v_term.reshape(B, 1), cost)
#     # return jnp.mean(cost)
#     return jnp.mean(jnp.sum(cost, axis=-1))