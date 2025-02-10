from typing import Tuple
import jax
import jax.numpy as jnp
import equinox as eqx
import mujoco.mjx as mjx
from jax import Array
from jaxtyping import PyTree
from mujoco.mjx import Data

from diff_sim.context.meta_context import Context
from diff_sim.simulate import controlled_simulate_fd as controlled_simulate

def loss_fn_policy_det(params: PyTree, static: PyTree, dxs:mjx.Data, ctx: Context, user_key: jnp.ndarray) -> tuple[
    jnp.ndarray, tuple[jnp.ndarray, mjx.Data, jnp.ndarray, jnp.ndarray]]:
    """
        Loss function for the direct analytical policy optimization problem given deterministic dynamics
        Args:
            params: PyTree, model parameters
            static: PyTree, static parameters
            dxs: mjx.Data
            ctx: Context, context object
            user_key: jnp.ndarray, random user_key for sub calls
        Returns:
            jnp.ndarray, loss value

        Notes:
            We compute the sum of the costs over the entire trajectory and average it over the batch
            loss = 1/B * sum_{b=1}^{B} sum_{t=1}^{T} cost(x_{b,t}, u_{b,t})
    """
    model = eqx.combine(params, static)
    dxs, x, ctrl, costs, _, terminated = controlled_simulate(dxs, ctx, model, user_key, ctx.cfg.nsteps) #shape: (B, T, 1)
    costs = jnp.sum(costs, axis=1)
    costs = jnp.mean(costs)
    return costs, (costs, dxs, terminated, x)


def loss_fn_policy_stoch(params: PyTree, static: PyTree, x_init: jnp.ndarray, ctx: Context, user_key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
        Loss function for the direct analytical policy optimization problem given stochastic dynamics.
        ** IF YOUR POLICY IS NOT STOCHASTIC (NO NOISE) OR SAMPLES = 1 THIS WILL BE IDENTICAL TO loss_fn_policy_det **
        Args:
            params: PyTree, model parameters
            static: PyTree, static parameters
            x_init: jnp.ndarray, initial state
            ctx: Context, context object
            user_key: jnp.ndarray, random user_key for sub calls
        Returns:
            jnp.ndarray, loss value

        Notes:
            We compute the expected sum of the costs over the entire trajectory and average it over the batch
            loss = 1/B * sum_{b=1}^{B} E[sum_{t=1}^{T} cost(x_{b,t}, u_{b,t})]
    """
    model = eqx.combine(params, static)
    _,_,costs,_,terminated = controlled_simulate(x_init, ctx, model, user_key)
    costs = costs.reshape(ctx.cfg.batch, ctx.cfg.samples, ctx.cfg.nsteps)
    sum_costs = jnp.sum(costs, axis=-1)
    exp_sum_costs = jnp.mean(sum_costs, axis=-1)
    costs = jnp.mean(exp_sum_costs)
    return costs, costs


def loss_fn_td_det(params: PyTree, static: PyTree, x_init: jnp.ndarray, ctx: Context, user_key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
        Loss function for the temporal difference value/policy optimization problem
        Args:
            params: PyTree, model parameters
            static: PyTree, static parameters
            x_init: jnp.ndarray, initial state
            ctx: Context, context object
            user_key: jnp.ndarray, random user_key for sub calls
        Returns:
            jnp.ndarray, loss value

        Notes:
            We compute the temporal difference loss over the entire trajectory and average it over the batch
            loss = 1/B * sum_{b=1}^{B} sum_{t=1}^{T} (v(x_{b,t}) - v(x_{b,t+1}) - c(x_{b,t}, u_{b,t}))^2
    """
    @jax.vmap
    def v_diff(x,t):
        v_seq = jax.vmap(model)(x, t)
        v0, v1 = v_seq[0:-1], v_seq[1:]
        return v0 - v1, v_seq[-1]

    @jax.vmap
    def td_cost(diff, term, cost):
        v_diff_cost = diff - cost[:-1]
        v_term_cost = term - cost[-1]
        return jnp.sum(jnp.square(v_diff_cost)) + jnp.square(v_term_cost)
    
    model = eqx.combine(params, static)
    dxs, x, _, costs, t, terminated = controlled_simulate(x_init, ctx, model, user_key)
    B, T, _ = x.shape
    diff, term = v_diff(x,t)
    traj_costs = jnp.mean(jnp.sum(costs, axis=-1))
    costs = td_cost(diff.reshape(B, T-1), term.reshape(B, 1), costs)
    return jnp.mean(costs), traj_costs


def loss_fn_td_stoch(params: PyTree, static: PyTree, x_init: jnp.ndarray, ctx: Context, user_key: jnp.ndarray) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
        Loss function for the temporal difference value/policy optimization problem
        IF YOUR POLICY IS NOT STOCHASTIC (NO NOISE) OR SAMPLES = 1 THIS WILL BE IDENTICAL TO loss_fn_td_det
        Args:
            params: PyTree, model parameters
            static: PyTree, static parameters
            x_init: jnp.ndarray, initial state
            ctx: Context, context object
            user_key: jnp.ndarray, random user_key for sub calls
        Return
            jnp.ndarray, loss value

        Notes:
            We compute the temporal difference loss over the entire trajectory and average it over the batch
            loss = 1/B * sum_{b=1}^{B} sum_{t=1}^{T} (v(x_{b,t}) - v(x_{b,t+1}) - c(x_{b,t}, u_{b,t}))^2
    """
    @jax.vmap
    def compute_values(x, t):
        return jax.vmap(model)(x, t)

    @jax.vmap
    def v_diff_stoch(values):
        v_average = jnp.mean(values, axis=0, keepdims=True)
        diff = values[:, 0:-1] - v_average[:, 1:]
        return diff, v_average[:, -1].flatten()

    @jax.vmap
    def td_cost_stoch(diff, term, cost):
        v_diff_cost = diff - cost[:, :-1]
        v_term_cost = term - jnp.mean(cost[:, -1])
        return jnp.mean(jnp.sum(jnp.square(v_diff_cost), axis=-1)) + jnp.square(v_term_cost)

    model = eqx.combine(params, static)
    dxs, x, _, costs, t,terminated = controlled_simulate(x_init, ctx, model, user_key)
    values = compute_values(x, t).reshape(ctx.cfg.batch, ctx.cfg.samples, ctx.cfg.nsteps)
    diff, term = v_diff_stoch(values) # shapes: (B, S, T-1), (B)
    costs  = costs.reshape(ctx.cfg.batch, ctx.cfg.samples, ctx.cfg.nsteps)
    traj_costs = jnp.mean(jnp.mean(jnp.sum(costs, axis=-1), axis=-1))
    costs = td_cost_stoch(diff, term, costs)
    return jnp.mean(costs), traj_costs

def loss_fn_target_det(params: PyTree, static: PyTree, x_init: jnp.ndarray, ctx: Context, user_key) -> tuple[jnp.ndarray, jnp.ndarray]:
    """
        Loss function for the target for fitted value iteration to learn value/policy
        Args:
            params: PyTree, model parameters
            static: PyTree, static parameters
            x_init: jnp.ndarray, initial state
            ctx: Context, context object
            user_key: jnp.ndarray, random user_key for sub calls
        Returns:
            jnp.ndarray, loss value

        Notes:
            We compute the target value over the entire trajectory and fit the value function over the batch
            loss = 1/B * sum_{b=1}^{B} sum_{t=1}^{T} (v(x_{b,t}) - y_{b,t})^2
    """
    @jax.vmap
    def cost(x, costs):
        pred = jax.vmap(model)(x, ctx.cfg.nsteps)
        targets = jnp.flip(costs)
        targets = jnp.flip(jnp.cumsum(targets))
        return jnp.sum(jnp.square(pred - targets))

    model = eqx.combine(params, static)
    dxs, x,_,costs,terminated =  controlled_simulate(x_init, ctx, model, user_key)
    traj_costs = jnp.mean(jnp.sum(costs, axis=1))
    return jnp.mean(cost(x,costs)), traj_costs
