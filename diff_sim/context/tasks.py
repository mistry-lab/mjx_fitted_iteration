# from diff_sim.context.cps import ctx as cp_ctx
# from diff_sim.context.di import ctx as di_ctx
# from diff_sim.context.shadow_hand import ctx as sh_ctx
from diff_sim.context.arm_push import ctx as ap_ctx
from diff_sim.context.ball_push import ctx as bp_ctx
# from diff_sim.context.point_mass import ctx as pm_ctx

ctxs = {
    # "double_integrator": di_ctx,
    # "shadow_hand": sh_ctx,
    "arm_push": ap_ctx,
    "ball_push": bp_ctx,

    # "point_mass": pm_ctx
}