# from diff_sim.context.cps import ctx as cp_ctx
from diff_sim.context.di import ctx as di_ctx
from diff_sim.context.shadow_hand import  ctx as sh_ctx
ctxs = {
    "double_integrator": di_ctx,
    # "cartpole_swing_up": cp_ctx,
    "shadow_hand": sh_ctx
}