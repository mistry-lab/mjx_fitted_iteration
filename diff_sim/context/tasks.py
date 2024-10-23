from diff_sim.context.cps import ctx as cp_ctx
from diff_sim.context.di import ctx as di_ctx
from diff_sim.context.shadow_hand import  ctx as sh_ctx
from diff_sim.context.simple_push import ctx as sp_ctx
from diff_sim.context.planar_arm import ctx as pa_ctx
from diff_sim.context.single_arm import ctx as sa_ctx

ctxs = {
    "double_integrator": di_ctx,
    "cartpole_swing_up": cp_ctx,
    "shadow_hand": sh_ctx,
    "simple_push": sp_ctx,
    "planar_arm": pa_ctx,
    "single_arm": sa_ctx
}