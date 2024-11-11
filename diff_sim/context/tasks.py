# from diff_sim.context.cps import ctx as cp_ctx
# from diff_sim.context.di import ctx as di_ctx
# from diff_sim.context.shadow_hand import ctx as sh_ctx
from diff_sim.context.single_arm import ctx as sa_ctx
# from diff_sim.context.point_mass import ctx as pm_ctx

ctxs = {
    # "double_integrator": di_ctx,
    # "shadow_hand": sh_ctx,
    "single_arm": sa_ctx
    # "point_mass": pm_ctx
}