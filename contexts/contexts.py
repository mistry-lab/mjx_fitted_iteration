import os
from .cps import ctx as cp_ctx
from .di import ctx as di_ctx

try:
    # This works when __file__ is defined (e.g., in regular scripts)
    base_path = os.path.dirname(__file__)
except NameError:
    # Fallback to current working directory (e.g., in interactive sessions)
    base_path = os.getcwd()

ctxs = {
    "double_integrator": di_ctx,
    "cartpole_swing_up": cp_ctx
}