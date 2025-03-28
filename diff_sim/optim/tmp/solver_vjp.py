


import jax
import jax.numpy as jnp
from jax import custom_vjp, vmap, tree_map
 
@custom_vjp
def solve(mx, dx):
    """The forward call of your solver, which returns the updated dx."""
    return solve_impl(mx, dx)
 
 
# ------------------- Forward pass definition ----------------------------------
def solve_fwd(mx, dx):
    """Forward pass: just call the solver. Save (mx, dx) for backward."""
    dx_out = solve_impl(mx, dx)
    residual = (mx, dx)  # store for backward
    return dx_out, residual
 
 
# ------------------- Backward pass definition ---------------------------------
def solve_bwd(residual, dx_out_bar):
    """
    Backward pass for 'solve' via finite difference.
 
    Args:
      residual: (mx, dx) from solve_fwd.
      dx_out_bar: PyTree shaped like dx_out (the solver outputs) representing
                  the incoming adjoint (d(Loss)/d(dx_out)).
 
    Returns:
      dmx, ddx: partial derivatives of Loss w.r.t mx and dx.
                We will produce dmx = None,
                and ddx a PyTree of the same shape as dx.
    """
    mx, dx_in = residual

    # Note:
    # ----
    # The following code can be used to convert float0 leaves to zeros in the cotangent 'dx_out_bar' 
    # Useful if iterativing over all field of dx_out_bar
    # def map_g_to_dinput(diff_tree, grad_tree):
    #     def fix_leaf(d_leaf, g_leaf):
    #         if jax.dtypes.result_type(g_leaf) == jax.dtypes.float0:
    #             return jnp.zeros_like(d_leaf)
    #         else:
    #             return g_leaf

    #     return jax.tree_map(fix_leaf, diff_tree, grad_tree)
    
    # dx_out_bar_mapped = map_g_to_dinput(dx_in, dx_out_bar)

    # --------------------------------------------------------------------------
    # 1) Build a single function that, given a small perturbation direction
    #    "delta_dx" in the shape of dx, returns the *directional derivative*
    #    of dx_out in that direction.
    # --------------------------------------------------------------------------
    eps = 1e-7 # for type Float64
 
    def apply_perturbation(dx, delta, sign):
        """Returns new dx with ± eps * delta for qpos, qvel, ctrl; zeros for others."""
        return dx.replace(
            qacc_smooth=dx.qacc_smooth + sign * eps * delta.qacc_smooth,
            qacc_warmstart=dx.qacc_warmstart + sign * eps * delta.qacc_warmstart
            # if dx has more fields, typically we do not perturb them => unchanged
        )
 
    def directional_dot(delta_dx):
        """
        Compute [d/d(delta_dx) solve_impl(mx, dx_in + eps * delta_dx)] dot dx_out_bar.
        This is effectively  ( J * delta_dx ) dot dx_out_bar = dx_out_bar^T ( J delta_dx ).
        By FD, we approximate J delta_dx as [solve(mx, dx+...) - solve(mx, dx-...)] / (2 eps).
        """
        # plus/minus
        dx_plus  = apply_perturbation(dx_in, delta_dx, +1.0)
        dx_minus = apply_perturbation(dx_in, delta_dx, -1.0)

        out_plus  = solve_impl(mx, dx_plus)
        out_minus = solve_impl(mx, dx_minus)

        # difference
        out_diff  = tree_map(lambda p, m: (p - m) / (2.0 * eps), out_plus, out_minus)

        return out_diff

 
    # ----------------------------------------------------------------------------------
    # 2) Build a 'batched' set of basis directions for qacc_smooth, qacc_warmstart .etc
    #    or any input to solve() function.
    #    Then we do a single vmap to get directional_dot for *each* basis direction.
    # -----------------------------------------------------------------------------

    # Note:
    # -----
    # The following code does not work as shape_ is not know at compiled time --> use mx. instead
    # def basis_like(x):
    #     shape_ = x.shape
    #     shape_ = mx.nv
    #     size_  = jnp.prod(jnp.array(shape_))
    #     eye_   = jnp.eye(size_, dtype=x.dtype)
    #     return eye_.reshape((size_,) + shape_)
 
    # Build "batched" directions for qacc_smooth and qacc_warmstart
    qacc_smooth_bases = jnp.eye(mx.nv, dtype=dx_in.qacc_smooth.dtype)
    qacc_warmstart_bases = jnp.eye(mx.nv, dtype=dx_in.qacc_warmstart.dtype)

    def build_perturbation(qacc_smooth ,qacc_warmstart):
        return dx_in.replace(qacc_smooth=qacc_smooth, qacc_warmstart= qacc_warmstart)
 
    def fd_qacc_smooth(qacc_smooth_dir):
        delta = build_perturbation(qacc_smooth_dir, jnp.zeros(mx.nv))
        return directional_dot(delta)
    
    def fd_qacc_warmstart(qacc_warmstart_dir):
        delta = build_perturbation(jnp.zeros(mx.nv), qacc_warmstart_dir)
        return directional_dot(delta)
    
    # Compute sensitivities.
    qacc_smooth_sensitivity = vmap(fd_qacc_smooth)(qacc_smooth_bases)
    qacc_warmstart_sensitivity = vmap(fd_qacc_warmstart)(qacc_warmstart_bases)

    # Note: 
    # -----
    # To compute the perturbation over the entire dx_in, run the following code :
    # qacc_smooth = jax.tree.reduce(lambda acc,x: acc + x, jax.tree_util.tree_map(lambda g,x: jnp.dot(x,g), 
    #                               dx_out_bar_mapped, qacc_smooth_sensitivity))
    # This returns an error as some of them include higher dimension (5x6) matrices for example.
    # Hence with a perturbation arund 3 axis, (3x5x6) x (5x6) returns an error.

    fields = ["qacc", "qacc_smooth", "qacc_warmstart", "qfrc_constraint", "efc_force", "ctrl"]
    qacc_smooth = sum(jnp.dot(getattr(qacc_smooth_sensitivity, f), getattr(dx_out_bar, f)) for f in fields)
    qacc_warmstart = sum(jnp.dot(getattr(qacc_warmstart_sensitivity, f), getattr(dx_out_bar, f)) for f in fields)
    
    dx_out_bar = dx_out_bar.replace(qacc_smooth=qacc_smooth, qacc_warmstart=qacc_warmstart)
    # jax.debug.breakpoint()
 
    # Return derivative wrt mx (None) and wrt dx_in (dx_in_bar).
    return (None, dx_out_bar)
 
 
# Attach the definitions to finalize the custom VJP:
solve.defvjp(solve_fwd, solve_bwd)